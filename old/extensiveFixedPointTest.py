#! /usr/bin/env python3

# extensive test of old fixed point library

import numpy as np

from fixpoint import FixedPoint

halfRange = (np.arange(2 ** 15 - 1) - (2 ** 14 - 1)) / (2 ** 8)

A = FixedPoint.fromFloats(halfRange, np.int16, 8)

for floatB in halfRange:
    if int(floatB) == floatB:
        print(floatB)
    B = FixedPoint.fromFloats(floatB, np.int16, 8)
    C = A + B
    D = A - B
    assert np.all(C.toFloats() == halfRange + floatB)
    assert np.all(D.toFloats() == halfRange - floatB)

sqrtRange = (np.arange(181 * 2 + 1) - 181) / 16

A = FixedPoint.fromFloats(sqrtRange, np.int16, 8)

for floatB in sqrtRange:
    if int(floatB) == floatB:
        print(floatB)
    B = FixedPoint.fromFloats(floatB, np.int16, 8)
    C = A * B
    assert np.all(C.toFloats() == sqrtRange * floatB)
