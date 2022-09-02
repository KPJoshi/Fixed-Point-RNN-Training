#! /usr/bin/env python3

# generates LUTs
# only relevant to small bitwidths which are not enough for training

from __future__ import annotations
import numpy as np
import warnings

# settings
SAFEMODE = True
dtype = np.int16
etype = np.int32
shift = 12
rounding = 'nearest'
# rounding = 'stochastic'

# init
warnings.filterwarnings('ignore', r'.*encountered.*')
bits = np.iinfo(dtype).bits
fxpInputs = np.arange(2 ** bits, dtype = dtype)
floatInputs = np.array(fxpInputs, dtype = float) / (2 ** shift)
floatMax = np.max(floatInputs)
if SAFEMODE:
    # use etype as dtype as safemode will often use it for overflow check
    dtype = etype

# 1) replace inf, -inf, etc.
# 2) clip to max, min
# 3) convert to fixed point
def sanitizeClipConvert(arr: np.ndarray) -> None:
    arr[np.isnan(arr)] = 0
    arr = np.clip(arr, -floatMax, floatMax)
    shfArr = arr * (2 ** shift)
    if rounding == 'nearest':
        fxpArr = np.round(shfArr)
    elif rounding == 'stochastic':
        sign = np.sign(shfArr)
        shfArr *= sign
        fxpArr = shfArr.astype(np.int64)
        fracPart = shfArr - fxpArr
        fracBits = 52 - int(np.log2(floatMax))
        halfFracBits = fracBits >> 1
        fracPart = np.array(fracPart * (2 ** (halfFracBits * 2)), dtype = np.int64)
        lowerFrac = fracPart & (2 ** halfFracBits - 1)
        upperFrac = fracPart >> halfFracBits
        fxpArr += (upperFrac > lowerFrac)
        fxpArr *= sign
    else:
        raise NotImplemented
    return np.array(fxpArr, dtype = dtype)

# create LUTs
LUTs = {}
LUTs['sqrtLUT'] = sanitizeClipConvert(floatInputs ** .5)
LUTs['recipLUT'] = sanitizeClipConvert(floatInputs ** -1)
LUTs['rsqrtLUT'] = sanitizeClipConvert(floatInputs ** -.5)
LUTs['expLUT'] = sanitizeClipConvert(np.exp(floatInputs))
LUTs['logLUT'] = sanitizeClipConvert(np.log(floatInputs))
LUTs['tanhLUT'] = sanitizeClipConvert(np.tanh(floatInputs))
LUTs['dtanhLUT'] = sanitizeClipConvert(np.cosh(floatInputs) ** -2)
sigmoid = 1 / (1 + np.exp(-floatInputs))
LUTs['sigmoidLUT'] = sanitizeClipConvert(sigmoid)
LUTs['dsigmoidLUT'] = sanitizeClipConvert(sigmoid * (1 - sigmoid))

# write LUTs
np.savez_compressed('LUT{}_{}_{}.npz'.format('_safe' if SAFEMODE else '', bits, shift), **LUTs)

# print loading code
for key in LUTs.keys():
    print(key, '=', 'fxpLUTs[' + repr(key) + ']')
