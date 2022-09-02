# evaluates piecewise polynomials - replaced by versions in iterativeapprox and fxpTensor

import numpy as np

import fxpTensor

class PiecewisePolynomialEvaluator:
    fxp0 = fxpTensor.fxpFromFloats(0.)
    fxpM1 = fxpTensor.fxpFromFloats(-1.)

    def __init__(self, segmentMaxs, segmentDeltas, segmentCoeffs, negativeHandling):
        self.segments = len(segmentMaxs)
        self.segmentMaxs = np.array(segmentMaxs)
        self.fxpSegmentMaxs = fxpTensor.fxpFromFloats(self.segmentMaxs)
        self.segmentDeltas = np.array(segmentDeltas)
        self.fxpSegmentDeltas = fxpTensor.fxpFromFloats(self.segmentDeltas)
        self.segmentCoeffs = np.array(segmentCoeffs)
        self.fxpSegmentCoeffs = fxpTensor.fxpFromFloats(self.segmentCoeffs)
        self.negativeHandling = np.array(negativeHandling)
        self.fxpNegativeHandling = fxpTensor.fxpFromFloats(self.negativeHandling)

    def floatEval(self, x: np.ndarray) -> np.ndarray:
        negatives = x < 0
        x = np.abs(x)
        segmentIdx = np.zeros(x.shape, dtype=int)
        for idx, segmentMax in enumerate(self.segmentMaxs):
            segmentIdx[x <= segmentMax] = idx
        x += np.choose(segmentIdx, self.segmentDeltas)
        accumulator = np.zeros_like(x)
        for coeff in self.segmentCoeffs:
            accumulator = (accumulator * x) + np.choose(segmentIdx, coeff)
        accumulator[negatives] = self.negativeHandling[0] + accumulator[negatives] * self.negativeHandling[1]
        return accumulator

    def fixPtEval(self, x: np.ndarray) -> np.ndarray:
        x = fxpTensor.fxpFromFloats(x)
        negatives = fxpTensor.compare(x, self.fxp0) < 0
        if np.any(negatives):
            x[negatives] = fxpTensor.fxpMul(x[negatives], self.fxpM1)
        segmentIdx = np.zeros(x.shape, dtype = int) # assume first segment up to inf
        for idx, segmentMax in enumerate(self.fxpSegmentMaxs):
            leqMax = fxpTensor.compare(x, segmentMax) <= 0
            segmentIdx[leqMax] = idx
        deltaX = np.choose(segmentIdx, self.fxpSegmentDeltas)
        x = fxpTensor.fxpAdd(x, deltaX)
        accumulator = np.zeros_like(x)
        for coeff in self.fxpSegmentCoeffs:
            segmentCoeff = np.choose(segmentIdx, coeff)
            accumulator = fxpTensor.fxpAdd(fxpTensor.fxpMul(accumulator, x, rounding = fxpTensor.fxpNearestRoundBreakEven), segmentCoeff)
        if np.any(negatives):
            accumulator[negatives] = fxpTensor.fxpMul(accumulator[negatives], self.fxpNegativeHandling[1])
            accumulator[negatives] = fxpTensor.fxpAdd(accumulator[negatives], self.fxpNegativeHandling[0])
        return fxpTensor.fxpToFloats(accumulator)
