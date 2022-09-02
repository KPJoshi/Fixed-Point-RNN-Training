# old fixed point library with support for different shift for each variable

from __future__ import annotations
import logging
import numpy as np
import traceback

SAFEMODE = True

if SAFEMODE:
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(__name__)

class FixedPoint:
    def __init__(self, shiftedValues: np.ndarray, dtype: type, etype: type, shift: int) -> None:
        self.dtype = dtype
        self.etype = etype
        self.shift = shift
        self.shiftedValues = shiftedValues

    @staticmethod
    def fromFloats(values: np.ndarray, dtype: type, shift: int) -> FixedPoint:
        etype = {np.dtype('int8'): np.int16, np.int8: np.int16,
                 np.dtype('int16'): np.int32, np.int16: np.int32,
                 np.dtype('int32'): np.int64, np.int32: np.int64} [dtype]
        shiftedValues = np.array(np.round(values * (2 ** shift)), dtype=etype)
        if SAFEMODE:
            shiftedValues = FixedPoint.clampOverflow(shiftedValues, dtype)
        else:
            shiftedValues = np.array(shiftedValues, dtype=dtype)
        return FixedPoint(shiftedValues, dtype, etype, shift)

    def toFloats(self) -> np.ndarray:
        result = np.array(self.shiftedValues, dtype=float)
        return result / (2 ** self.shift)

    def __copy__(self) -> FixedPoint:
        return FixedPoint(self.shiftedValues, self.dtype, self.etype, self.shift)

    def __deepcopy__(self) -> FixedPoint:
        return FixedPoint(self.shiftedValues.copy(), self.dtype, self.etype, self.shift)

    def comparable(self, othr: FixedPoint) -> bool:
        return (self.dtype == othr.dtype and
                self.shift == othr.shift)

    def compare(self, othr: FixedPoint) -> np.ndarray:
        if SAFEMODE:
            assert self.comparable(othr)
        return ((self.shiftedValues < othr.shiftedValues) * -1 +
                (self.shiftedValues > othr.shiftedValues) * 1)

    @staticmethod
    def clampOverflow(arr: np.ndarray, dtype: type) -> np.ndarray:
        dtypeMax = np.iinfo(dtype).max
        if np.any((arr < -dtypeMax) | (arr > dtypeMax)):
            logger.warning('Clamped %d to ±%d', np.max(np.abs(arr)), dtypeMax)
            traceback.print_stack()
            arr = np.clip(arr, -dtypeMax, dtypeMax)
        return np.array(arr, dtype=dtype)

    def pseudoStochasticRound(self, arr: np.ndarray) -> np.ndarray:
        negative = arr < 0
        arr = np.abs(arr)
        upperBits = arr >> self.shift
        lowerBits = np.array(arr & ((2 ** self.shift) - 1), self.dtype)
        if (self.shift & 1) == 1:
            lowerBits >>= 1
        halfShift = self.shift >> 1
        upperHalf = lowerBits >> halfShift
        lowerHalf = lowerBits & ((2 ** halfShift) - 1)
        roundUp = upperHalf > lowerHalf
        upperBits += roundUp
        if SAFEMODE:
            upperBits = FixedPoint.clampOverflow(upperBits, self.dtype)
        else:
            upperBits = np.array(upperBits, self.dtype)
        upperBits[negative] *= -1
        return upperBits
    
    def nearestRoundBreakEven(self, arr: np.ndarray) -> np.ndarray:
        lowerBits = (arr & (2 ** (self.shift - 1) - 1))
        middleBits = (arr >> (self.shift - 1)) & 3
        roundUp = (middleBits == 3) | ((middleBits == 1) & (lowerBits > 0))
        result = (arr >> self.shift) + roundUp
        if SAFEMODE:
            result = FixedPoint.clampOverflow(result, self.dtype)
        else:
            result = np.array(result, self.dtype)
        return result

    rounding = nearestRoundBreakEven

    def __add__(self, othr: FixedPoint) -> FixedPoint:
        if SAFEMODE:
            assert self.comparable(othr)
            selfExtValues = np.array(self.shiftedValues, dtype=self.etype)
            othrExtValues = np.array(othr.shiftedValues, dtype=self.etype)
            resultExtValues = selfExtValues + othrExtValues
            resultShiftedValues = FixedPoint.clampOverflow(resultExtValues, self.dtype)
        else:
            resultShiftedValues = self.shiftedValues + othr.shiftedValues
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)

    def __sub__(self, othr: FixedPoint) -> FixedPoint:
        if SAFEMODE:
            assert self.comparable(othr)
            selfExtValues = np.array(self.shiftedValues, dtype=self.etype)
            othrExtValues = np.array(othr.shiftedValues, dtype=self.etype)
            resultExtValues = selfExtValues - othrExtValues
            resultShiftedValues = FixedPoint.clampOverflow(resultExtValues, self.dtype)
        else:
            resultShiftedValues = self.shiftedValues - othr.shiftedValues
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)

    def __neg__(self) -> FixedPoint:
        resultShiftedValues = -self.shiftedValues
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)

    def __mul__(self, othr: FixedPoint) -> FixedPoint:
        if SAFEMODE:
            assert self.comparable(othr)
        selfExtValues = np.array(self.shiftedValues, dtype=self.etype)
        othrExtValues = np.array(othr.shiftedValues, dtype=self.etype)
        doubleShiftedProduct = selfExtValues * othrExtValues
        resultShiftedValues = self.rounding(doubleShiftedProduct)
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)
    
    def T(self) -> FixedPoint:
        return FixedPoint(self.shiftedValues.T, self.dtype, self.etype, self.shift)

    def mm(self, othr: FixedPoint) -> FixedPoint:
        selfExtValues = np.array(self.shiftedValues, dtype=self.etype)
        othrExtValues = np.array(othr.shiftedValues, dtype=self.etype)
        doubleShiftedProduct = np.matmul(selfExtValues, othrExtValues)
        resultShiftedValues = self.rounding(doubleShiftedProduct)
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)

    # create FixedPoint array with provided choices from this array
    def choose(self, choices: np.ndarray) -> FixedPoint:
        resultShiftedValues = np.choose(choices, self.shiftedValues)
        return FixedPoint(resultShiftedValues, self.dtype, self.etype, self.shift)

    def simulateLUTEval(self, fn) -> FixedPoint:
        floatValues = self.toFloats()
        resultValues = fn(floatValues)
        return FixedPoint.fromFloats(resultValues, self.dtype, self.shift)
