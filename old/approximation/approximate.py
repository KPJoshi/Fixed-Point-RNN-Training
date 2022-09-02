#! /usr/bin/env python3

# old approximation code requiring manual segment creation
# could be more efficient in terms of no. of required segments for the same error guarantee as opposed to splitting in half
# also creates an error estimate graph
# mostly replacable by the iterative approximation script

import chaospy
import matplotlib.pyplot as plt
import numpy as np

# functions to approximate
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
def dSigmoid(x: float) -> float:
    nExp = np.exp(-x)
    return nExp / (1 + nExp) ** 2
def tanh(x: float) -> float:
    return np.tanh(x)
def dTanh(x: float) -> float:
    return np.cosh(x) ** -2
def rsqrt(x: float) -> float:
    return x ** -.5

# GPC settings
gpcOrder = 2
segments = [(0, .5), (.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8)] # S:3ULP@O2
# segments = [(0, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 8)] # T:4ULP@O2
gpcFunction = sigmoid

gpcModels = []
segmentOffsets = []
segmentCoeffs = []
for segment in segments:
    print('Approximating in segment', segment)
    isFinite = np.isfinite(segment[1])
    # prevent exponential blowup by recentering
    segment = np.array(segment, dtype=float)
    if isFinite:
        segmentOffset = np.mean(segment)
    else:
        segmentOffset = segment[0]
    segment -= segmentOffset
    # can use different dist here such as truncnormal, truncexp
    # gpcDist = chaospy.TruncNormal(*segment, -segmentOffset, 2)
    # if isFinite:
    #     gpcDist = chaospy.TruncExponential(segment[1], 2, segment[0])
    # else:
    #     gpcDist = chaospy.Exponential(2, segment[0])
    gpcDist = chaospy.Uniform(*segment)
    if isFinite:
        gpcExpansion = chaospy.generate_expansion(gpcOrder, gpcDist)
        gpcSamples, gpcWeights = chaospy.generate_quadrature(gpcOrder, gpcDist, rule='gaussian')
    else:
        gpcExpansion = chaospy.generate_expansion(0, gpcDist)
        gpcSamples, gpcWeights = chaospy.generate_quadrature(0, gpcDist, rule='gaussian')
    gpcEvals = gpcFunction(gpcSamples + segmentOffset)
    # hack because 1D expected but gpcEvals is 2D
    gpcEvals = gpcEvals[0]
    # debug info
    # print(gpcExpansion)
    # print(gpcSamples)
    # print(gpcWeights)
    # print(gpcEvals)
    gpcModel = chaospy.fit_quadrature(gpcExpansion, gpcSamples, gpcWeights, gpcEvals)
    gpcModelStr = str(gpcModel).replace('q0', '(x-{})'.format(segmentOffset))
    # print(gpcModelStr)
    # store data for later formatting
    gpcModels.append(gpcModel)
    segmentOffsets.append(-segmentOffset)
    if isFinite:
        segmentCoeffs.append(gpcModel.coefficients)
    else:
        segmentCoeffs.append(gpcModel.coefficients + [0]*gpcOrder)

# print in format used by other code
print('Polynomial approximation data:')
segmentMaxs = [segment[1] for segment in segments]
segmentMaxs.reverse()
print('segmentMaxs =', segmentMaxs)
segmentOffsets.reverse()
print('segmentDeltas =', segmentOffsets)
print('segmentCoeffs = [')
segmentCoeffs = np.flip(np.array(segmentCoeffs).T, axis = (0, 1))
for coeffs in segmentCoeffs:
    print('    ', list(coeffs), ',', sep='')
print(']')

print('Plotting error estimate graph...')
xRange = (segments[0][0], min(25, segments[-1][1]))
X = np.linspace(xRange[0], xRange[1], 2 ** 11, endpoint = False)
Yp = gpcFunction(X)
Ya = np.empty_like(Yp)
for xIdx, x in enumerate(X):
    for sIdx, segment in enumerate(segments):
        if segment[1] >= x:
            Ya[xIdx] = gpcModels[sIdx](x + segmentOffsets[len(segments) - 1 - sIdx])
            break
EY = Yp - Ya

fig, ax = plt.subplots()
ax.plot(X, Yp, label='golden', color='green')
ax.plot(X, Ya, label='poly', color='yellow', linestyle='dotted')
ax.plot(X, EY * (2 ** 12), label='err * 2^12', color='red')
ax.legend()
ax.grid(True, which='both')
ax.axhline(y=0, color='black')
ax.axvline(x=0, color='black')
plt.savefig('error.png')
