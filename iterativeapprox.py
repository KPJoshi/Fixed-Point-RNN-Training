#! /usr/bin/env python3

from __future__ import annotations
import chaospy
import numpy as np

# used to evaluate the approximation with the actual fixed point library
import fxpTensor

# -------------
# main settings
# -------------

# function to approximate - you can add more, but you will also have to add func and negativeHandling below
funcName = 'sigmoid'
# funcName = 'tanh'
# funcName = 'rsqrt'

# controls range of approximation
intbits = 5

# error targets
targetActualError = 3 * 2 ** -12 # max allowable actual error of approximation
targetULPError = np.inf # max allowable ULP error of approximation - must be an integer

# the order of the polynomial approximation used on each segment
order = 2

# -------------------
# additional settings
# -------------------

# creates initial segment and tests for entire range of approximation
# exhaustive testing may be improactial for larger range of approximation - in that case, less tests are used
shift = fxpTensor.shift
initialSegment = (0., 2. ** intbits)
tests = np.linspace(- 2. ** intbits, 2 ** intbits, 2 ** min(shift + intbits + 1, 26) + 1)
print('ULPs between tests:', (tests[1]-tests[0]) * (2 ** shift))
if funcName == 'rsqrt':
    initialSegment = (1., 4.)
    tests = np.arange(3 * 2 ** shift) / (2 ** shift) + 1

# defines the actual function to approximate and how f(x) can be transformed into f(-x) - you can add new functions here
if funcName == 'sigmoid':
    func = lambda x: 1 / (1 + np.exp(-x))
    negativeHandling = lambda x: fxpTensor.fxpSub(fxpTensor.fxpFromFloats(1.), x)
elif funcName == 'tanh':
    func = np.tanh
    negativeHandling = lambda x: -x
elif funcName == 'rsqrt':
    func = lambda x: x ** -.5
    negativeHandling = None
else:
    raise ValueError

# ---------
# main code
# ---------

# creates GPC approximation for a given segment
# returns the coefficients of the polynomial approximation and the normalization constant
# partially normalizing the segment by shifting it to be around 0 increases quality of approximation
def createGPCApprox(segment: tuple) -> tuple:
    # handling for infinite segments - not used right now
    isFinite = np.isfinite(segment[1])
    segment = np.array(segment, dtype=float)
    if isFinite:
        # for finite segments, create approximation around center of segment
        delta = np.mean(segment)
        delta = np.round(delta * (2 ** shift)) / (2 ** shift)
    else:
        delta = segment[0]
    # shift segment min/max
    shfSegment = segment - delta
    if isFinite:
        # for finite segments, assume that the approximation will be evaluated uniformly over the segment
        # a truncated exponential distribution over the segment can be used instead
        gpcDist = chaospy.Uniform(*shfSegment)
        gpcOrder = order
    else:
        # for infinite segments, assume that approximation will be evaluated mostly around segment lower bound
        gpcDist = chaospy.Exponential() # scale, shift automatically 1, 0
        gpcOrder = 0 # forces a constant approximation
    # generate orthogonal polynomials
    gpcExpansion = chaospy.generate_expansion(gpcOrder, gpcDist)
    # generate evaluation points for numerical integration via gaussian quadrature
    gpcSamples, gpcWeights = chaospy.generate_quadrature(gpcOrder, gpcDist, rule='gaussian')
    # evaluate function at optimal evaluation points
    gpcEvals = func(gpcSamples + delta)
    gpcEvals = gpcEvals[0] # hack because 1D expected but gpcEvals is 2D
    # calculate polynomial approximation using what is essentially a fourier transform
    gpcModel = chaospy.fit_quadrature(gpcExpansion, gpcSamples, gpcWeights, gpcEvals)
    # return polynomial coefficients and normalization constant
    gpcCoeff = gpcModel.coefficients
    gpcCoeff.reverse()
    return gpcCoeff, -delta

# recursively breaks segments in half to ensure error bounds are met
class SegmentApprox:
    # evaluates approximation using fixed point library
    def fixPtEval(self, x: np.ndarray) -> np.ndarray:
        x = fxpTensor.fxpFromFloats(x) # x (input)
        delta = fxpTensor.fxpFromFloats(self.delta) # segment delta (calculated by createGPCApprox)
        coeffs = fxpTensor.fxpFromFloats(np.array(self.coeffs)) # polynomial coeffs (calculated by createGPCApprox)
        self.coeffs = fxpTensor.fxpToFloats(coeffs) # store the fixed point represented coeffs back for printing
        # store location of negatives and calculate absolute value
        negatives = x < 0
        if np.any(negatives):
            x[negatives] = -x[negatives]
        # shift x
        x = fxpTensor.fxpAdd(x, delta)
        # calculate approximate output via multiply-adds of shifted x and the coefficients
        accumulator = coeffs[0]
        for coeff in coeffs[1:]:
            temp = fxpTensor.fxpMul(accumulator, x, rounding = fxpTensor.fxpNearestRoundBreakEven)
            accumulator = fxpTensor.fxpAdd(temp, coeff)
        # adjust for negative x
        if np.any(negatives):
            accumulator[negatives] = negativeHandling(accumulator[negatives])
        # convert back to float and return
        return fxpTensor.fxpToFloats(accumulator)

    # create and store approximation for this segment, splitting segment if approximation is not good enough
    def __init__(self, segment):
        self.segment = segment
        # create approximation
        print('Approximating in segment', segment)
        self.coeffs, self.delta = createGPCApprox(segment)
        # calculate subset of tests relevant to this segment
        segmentTests = tests[(np.abs(tests) <= segment[1])]
        if segment[0] > 0:
            segmentTests = segmentTests[(np.abs(segmentTests) > segment[0])]
        # run the tests
        goldenVals = func(segmentTests)
        fixPntVals = self.fixPtEval(segmentTests)
        # calculate actual error
        errors = goldenVals - fixPntVals
        maxAbsError = np.max(np.abs(errors))
        # calculate ULP error
        fxpGoldenVals = np.round(goldenVals * (2 ** shift)).astype(int)
        ULPErrors = fxpGoldenVals - (fixPntVals * (2 ** shift)).astype(int)
        maxAbsULPError = np.max(np.abs(ULPErrors))
        # if error above threshold, split segment in half and recursively approximate
        if maxAbsError > targetActualError or maxAbsULPError > targetULPError:
            print('Error above threshold; splitting segment')
            child0 = SegmentApprox((segment[0], -self.delta))
            child1 = SegmentApprox((-self.delta, segment[1]))
            self.children = [child0, child1]
        else:
            print('Error within threshold')
            self.children = None

    # return all segments and sub-segments in order
    def toList(self) -> list[SegmentApprox]:
        if self.children is None:
            return [self]
        else:
            return self.children[0].toList() + self.children[1].toList()

# recursively splits and approximates starting from initial segment
approximations = SegmentApprox(initialSegment)
# retrieve all segment approximations
approximations = approximations.toList()
# print in format used by fixed point library
approximations.reverse()
segmentMaxs = [approx.segment[1] for approx in approximations]
print('segmentMaxs =', segmentMaxs) # upper bounds of each segment
segmentDeltas = [approx.delta for approx in approximations]
print('segmentDeltas =', segmentDeltas) # normalization constants
print('segmentCoeffs = [') # polynomial coefficients, from highest to lowest degree
for idx in range(order + 1):
    segmentCoeffs = [approx.coeffs[idx] for approx in approximations]
    print('\t', segmentCoeffs, ',', sep = '')
print(']')

