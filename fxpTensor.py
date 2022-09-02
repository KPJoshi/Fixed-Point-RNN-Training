from __future__ import annotations
import builtins
import numpy as np
import sys
import traceback

# settings
SAFEMODE = True # check for and clip overflows
dtype = np.int32 # base data type
etype = np.int64 # extended data type - temporarily used for multiplies
shift = 24 # frac bits

# init
dtypeMax = np.iinfo(dtype).max # max value of base data type
bits = np.iinfo(dtype).bits # bits in base data type
fxpEps = 2 ** -shift # min difference between representable values
fxpMax = dtypeMax * fxpEps # max representable value
if SAFEMODE:
    # use etype as dtype as safemode will often use it for overflow check
    dtype = etype

####################
# fixpoint functions
####################

# dict to keep track of overflows - used in safe mode
overflowTracebacks = {}

# display overflow statistics (# overflows, max value) along with stacktrace of where it occurred
def printOverflowStatistics() -> None:
    for overflowCount, overflowMax, overflowTrace in overflowTracebacks.values():
        print(overflowCount, 'overflow(s) up to', overflowMax, ':', file = sys.stderr)
        print(overflowTrace, file = sys.stderr)
        print(file = sys.stderr)

# detect and clip overflows, store statistics of overflow
def fxpClampOverflow(arr: np.ndarray) -> np.ndarray:
    arrMin, arrMax = np.min(arr), np.max(arr)
    # check if overflow has occurred
    if arrMin < -dtypeMax or arrMax > dtypeMax:
        # retrieve stack trace where overflow occurred
        stackSummary = traceback.extract_stack()
        tracebackStr = '\n'.join([str(frame.lineno) + '\t' + frame.line.strip() for frame in stackSummary])
        tracebackHash = hash(tracebackStr)
        # store overflow stats for future lookup
        if tracebackHash not in overflowTracebacks:
            overflowTracebacks[tracebackHash] = [0, dtypeMax, tracebackStr]
        tracebackStats = overflowTracebacks[tracebackHash]
        tracebackStats[0] += 1
        tracebackStats[1] = max(tracebackStats[1], abs(arrMax), abs(arrMin))
        # bring value into representable range
        arr = np.clip(arr, -dtypeMax, dtypeMax)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
    return arr

# float -> fxp
def fxpFromFloats(values: np.ndarray) -> np.ndarray:
    shValues: np.ndarray = np.round(values / fxpEps)
    shValues = shValues.astype(dtype)
    if SAFEMODE:
        shValues = fxpClampOverflow(shValues)
    return shValues

fxpOne = fxpFromFloats(1.) # used by many functions

# fxp -> float
def fxpToFloats(shValues: np.ndarray) -> np.ndarray:
    values = shValues.astype(float)
    return values * fxpEps

# return -1 if a<b, 0 if a=b, and 1 if a>b
def compare(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a < b) * -1 + (a > b) * 1)

# used by pseudo-stochastic round
assert shift & 1 == 0 # for efficient pseudo stochastic round
PSRHalfShift = shift // 2
PSRHalfFracMask = (2 ** PSRHalfShift) - 1

# pseudo-stochastic rounding - better than nearest rounding in most cases
# see https://arxiv.org/abs/2009.13108
def fxpPseudoStochasticRound(arr: np.ndarray) -> np.ndarray:
    # calculate bas
    sign = np.sign(arr)
    arr = arr * sign
    # do round
    intPart: np.ndarray = arr >> shift
    roundUp = ((arr >> PSRHalfShift) & PSRHalfFracMask) > (arr & PSRHalfFracMask)
    intPart += roundUp
    # restore sign
    intPart *= sign
    if SAFEMODE:
        intPart = fxpClampOverflow(intPart)
    else:
        intPart = intPart.astype(dtype)
    return intPart

# like above, but always does clip - could just be made the default
def fxpSafePseudoStochasticRound(arr: np.ndarray) -> np.ndarray:
    sign = np.sign(arr)
    arr = arr * sign
    intPart: np.ndarray = arr >> shift
    roundUp = ((arr >> PSRHalfShift) & PSRHalfFracMask) > (arr & PSRHalfFracMask)
    intPart += roundUp
    intPart *= sign
    intPart = np.clip(intPart, -dtypeMax, dtypeMax)
    if not SAFEMODE:
        intPart = intPart.astype(dtype)
    return intPart

# round to nearest, break towards even - standard rounding used by numpy, etc.
# only useful when approximating sigmoid, tanh, rsqrt, etc.
# used elsewhere, it causes loss of model accuracy
def fxpNearestRoundBreakEven(arr: np.ndarray) -> np.ndarray:
    lowerBits = (arr & (2 ** (shift - 1) - 1))
    middleBits = (arr >> (shift - 1)) & 3
    roundUp = (middleBits == 3) | ((middleBits == 1) & (lowerBits > 0))
    result: np.ndarray = (arr >> shift) + roundUp
    if SAFEMODE:
        result = fxpClampOverflow(result)
    else:
        result = result.astype(dtype)
    return result

# add 2 or more fixpoints - just use Python's sum function
def fxpAdd(*args: np.ndarray) -> np.ndarray:
    result = builtins.sum(args)
    if SAFEMODE:
        result = fxpClampOverflow(result)
    return result

# subtract 2 fixpoints
def fxpSub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = a - b
    if SAFEMODE:
        result = fxpClampOverflow(result)
    return result

# multiply 2 fixpoints by sign-extending, multiplying, and rounding
def fxpMul(a: np.ndarray, b: np.ndarray, rounding = fxpPseudoStochasticRound) -> np.ndarray:
    if SAFEMODE:
        aExt, bExt = a, b
    else:
        aExt = a.astype(etype)
        bExt = b.astype(etype)
    resultExt = aExt * bExt
    result = rounding(resultExt)
    return result

# like np.sum operation
def fxpSum(a: np.ndarray, axis = None) -> np.ndarray:
    result = np.sum(a, axis = axis)
    if SAFEMODE:
        result = fxpClampOverflow(result)
    if result.dtype != dtype:
        result = result.astype(dtype)
    return result

# like np.matmul, flow similar to multiply
def fxpMM(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if SAFEMODE:
        aExt, bExt = a, b
    else:
        aExt = a.astype(etype)
        bExt = b.astype(etype)
    resultExt = np.matmul(aExt, bExt)
    result = fxpPseudoStochasticRound(resultExt)
    return result

# safe fused multiply add
# multiplies multiple matrices (standard matrix mult or pointwise mult), adds them, clips, and rounds
# pointwise and matrix multiplies can be interspersed (this much flexibility is probably unnecessary)
def fxpSFMAdd(args: list[np.ndarray], ops: list) -> np.ndarray:
    assert len(args) == len(ops) * 2
    if SAFEMODE:
        extArgs = args
    else:
        extArgs = []
        for arg in args:
            extArgs.append(arg.astype(etype))
    extResult = np.zeros((args[0].shape[0], args[1].shape[1]), dtype = etype)
    for idx, op in enumerate(ops):
        extResult += op(extArgs[idx * 2], extArgs[idx * 2 + 1])
    result = fxpSafePseudoStochasticRound(extResult)
    return result

# prints info about the fixedpoint array
def fxpDebug(arr: np.ndarray, name: str) -> None:
    fxpMin, fxpMax = np.min(arr), np.max(arr)
    floatMin, floatMax = fxpMin * fxpEps, fxpMax * fxpEps
    print(name, arr.shape, arr.dtype, [fxpMin, fxpMax], [floatMin, floatMax])

###############
# LUT interface
###############

# simulate a LUT by converting to float, applying function, and converting back
def fxpSimLUT(a: np.ndarray, fn) -> np.ndarray:
    floatValues = fxpToFloats(a)
    resultValues = fn(floatValues)
    return fxpFromFloats(resultValues)

# reads actual pregenerated LUTs - only useful for small bitwidths, not much perf gain compared to simLUT above
# lutFile = 'data/LUT{}_{}_{}.npz'.format('_safe' if SAFEMODE else '', bits, shift)
# fxpLUTs = np.load(lutFile)
# sqrtLUT = fxpLUTs['sqrtLUT']
# recipLUT = fxpLUTs['recipLUT']
# rsqrtLUT = fxpLUTs['rsqrtLUT']
# expLUT = fxpLUTs['expLUT']
# logLUT = fxpLUTs['logLUT']
# tanhLUT = fxpLUTs['tanhLUT']
# dtanhLUT = fxpLUTs['dtanhLUT']
# sigmoidLUT = fxpLUTs['sigmoidLUT']
# dsigmoidLUT = fxpLUTs['dsigmoidLUT']

# record which parts of the LUTs are used more
# could be useful for deciding which parts of a piecewise polynomial approximation need more precision
LUTUsageStats = {}
def recordLUTUsageStats(arr: np.ndarray, name: str):
    if name not in LUTUsageStats:
        LUTUsageStats[name] = np.zeros(2 ** bits, dtype = int)
    usage = np.bincount(arr.flatten().astype(np.uint16))
    usage.resize(2 ** bits)
    LUTUsageStats[name] += usage

####################################
# piecewise polynomial approximation
####################################

# used by the rsqrt, sigmoid, and tanh approximations
# stores and evaluates a piecewise polynomial
class PiecewisePolynomialEvaluator:
    def __init__(self, segmentMaxs, segmentDeltas, segmentCoeffs, negativeHandling):
        self.segments = len(segmentMaxs) # num segments
        self.segmentMaxs = fxpFromFloats(np.array(segmentMaxs)) # upper limits of segments
        self.segmentDeltas = fxpFromFloats(np.array(segmentDeltas)) # normalization factors of segments
        self.segmentCoeffs = fxpFromFloats(np.array(segmentCoeffs)) # polynomial coefficients
        self.negativeHandling = negativeHandling # how to handle negative inputs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # calculate absolute value
        negatives = x < 0
        x = np.abs(x)
        # identify segment containing input by comparing against segment boundaries
        segmentIdx = np.zeros(x.shape, dtype = int)
        for idx, segmentMax in enumerate(self.segmentMaxs):
            leqMax = compare(x, segmentMax) <= 0
            segmentIdx[leqMax] = idx
        # normalize input
        deltaX = np.choose(segmentIdx, self.segmentDeltas)
        x = fxpAdd(x, deltaX)
        # do multiply-adds with selected segment coefficients and input
        # use round to nearest, seems to be optimal in this situation
        accumulator = np.zeros_like(x)
        for coeff in self.segmentCoeffs:
            segmentCoeff = np.choose(segmentIdx, coeff)
            accumulator = fxpAdd(fxpMul(accumulator, x, rounding = fxpNearestRoundBreakEven), segmentCoeff)
        # handle negative inputs
        if np.any(negatives):
            accumulator[negatives] = self.negativeHandling(accumulator[negatives])
        return accumulator

##########################
# tensor-related functions
##########################

# modeled after pytorch's Tensor datatype
class Tensor:
    tempCounter = 0 # automatic naming of tensors - could be useful for autogenerating code

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        # can initialize either with floats or directly with fixpoints
        if data.dtype == float:
            self.data = fxpFromFloats(data)
        elif data.dtype == dtype:
            self.data = data
        else:
            raise ValueError
        # gradient of this tensor
        self.grad: np.ndarray = None
        # backward of function used to calculate this tensor
        self.bwdCls: Func = None
        # context to be used with backward function
        self.bwdCtx: list = None
        # tensors used to calculate this tensor
        self.bwdArgs: list[Tensor] = None
        # name of this tensor
        if name is not None:
            self.name = name
        else:
            self.name = 'tt_' + str(Tensor.tempCounter)
            Tensor.tempCounter += 1

    # topological sort of computation DAG starting from this node
    # necessary for correctly performing backward pass
    def topoSort(self) -> list[Tensor]:
        exploredSet: set[Tensor] = set()
        calcOrder: list[Tensor] = []
        # recursive DFS function
        def dfs(node: Tensor) -> None:
            # first explore children if not already explored
            if node.bwdArgs:
                for input in node.bwdArgs:
                    if input not in exploredSet:
                        dfs(input)
            # mark as explored and add to calcOrder
            exploredSet.add(node)
            calcOrder.append(node)
        dfs(self)
        return calcOrder

    # efficient backward pass starting from this root node
    def backward(self) -> None:
        # must start from scalar
        assert self.data.size == 1
        # get calculation order via DFS
        gradCalcOrder = self.topoSort()
        # reverse to get proper order - start backward pass from loss
        gradCalcOrder.reverse()
        # zero all gradients
        for node in gradCalcOrder:
            node.grad = np.zeros_like(node.data)
        # gradient of root is 1
        self.grad = fxpOne.reshape(self.data.shape)
        # backward pass in topological order
        for node in gradCalcOrder:
            if node.bwdArgs:
                inputs_grad = node.bwdCls.bwd(node.bwdCtx, node.grad)
                for input, input_grad in zip(node.bwdArgs, inputs_grad):
                    input.grad += input_grad

    # update parameter tensor
    def update(self, delta: np.ndarray) -> None:
        self.data = fxpAdd(self.data, delta)
        # reset other variables
        self.grad = self.bwdCls = self.bwdCtx = self.bwdArgs = None

    def __str__(self) -> str:
        return str(fxpToFloats(self.data))

    # basic operator overloading: +, -, *, array slicing

    def __add__(self, othr: Tensor) -> Tensor:
        return AddFunc.apply(self, othr)

    def __sub__(self, othr: Tensor) -> Tensor:
        return SubFunc.apply(self, othr)

    def __mul__(self, othr: Tensor) -> Tensor:
        return MulFunc.apply(self, othr)

    def __getitem__(self, key) -> Tensor:
        return GetItemFunc.apply(self, key = key)

# ADAM optimizer
class AdamOptimizer:
    useLUT = False
    # approximation for reciprocal of square root in range [1,4]
    rsqrtPPApprox = PiecewisePolynomialEvaluator(
        [4.0, 3.25, 2.5, 2.125, 1.75, 1.375, 1.1875],
        [-3.625, -2.875, -2.3125, -1.9375, -1.5625, -1.28125, -1.09375],
        [
            [0.0150604248046875, 0.0269622802734375, 0.0462493896484375, 0.0720672607421875, 0.1236572265625, 0.2022857666015625, 0.3007049560546875],
            [-0.0727386474609375, -0.1032257080078125, -0.1425323486328125, -0.1860504150390625, -0.25738525390625, -0.345458984375, -0.438323974609375],
            [0.5252227783203125, 0.5897674560546875, 0.6575927734375, 0.718414306640625, 0.8000030517578125, 0.883453369140625, 0.9561767578125],
        ],
        None # negatives not valid
    ).evaluate
    @staticmethod
    def fxpRsqrt(arr: np.ndarray) -> np.ndarray:
        # sanity check
        assert np.all(arr > 0)
        # range reduction to [1,4)
        shiftedArr = arr.copy()
        rrshift = np.zeros(arr.shape, dtype = int)
        while True:
            below1 = shiftedArr < fxpOne
            if not np.any(below1):
                break
            shiftedArr[below1] <<= 2
            rrshift[below1] += 1
        # evaluate approximation
        shiftedRsqrt = AdamOptimizer.rsqrtPPApprox(shiftedArr)
        # undo range reduction
        rsqrt = shiftedRsqrt << rrshift # no overflow - auto-promotion to int
        # clip important - often overflows
        rsqrt = np.clip(rsqrt, 0, dtypeMax).astype(dtype)
        return rsqrt

    def __init__(self, params: list[Tensor], a = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-8):
        self.params = params
        self.a = a # alpha parameter
        self.b1 = fxpFromFloats(min(b1, 1 - fxpEps)) # beta1 parameter - max 1-eps
        self.b2 = fxpFromFloats(min(b2, 1 - fxpEps)) # beta2 parameter - max 1-eps
        self.e = fxpFromFloats(max(e, fxpEps)) # epsilon (for preventing divide by 0) - min eps
        self.b1t = 1. # b1 ^ t
        self.b2t = 1. # b2 ^ t
        self.flb1 = fxpToFloats(self.b1) # float version of b1 - calculated from fixedpoint to ensure equality
        self.flb2 = fxpToFloats(self.b2) # float version of b2 - calculated from fixedpoint to ensure equality
        self.m = [] # mu - first moment of gradient
        self.n = [] # nu - second moment of gradient
        for param in params:
            self.m.append(np.zeros_like(param.data))
            self.n.append(np.zeros_like(param.data))

    def step(self) -> None:
        # update hyperparams (in float - public info) and convert to fixpoint
        self.b1t *= self.flb1
        self.b2t *= self.flb2
        at = self.a * np.sqrt(1 - self.b2t) / (1 - self.b1t)
        at = fxpFromFloats(max(at, fxpEps))
        # update moments and param
        for idx, param in enumerate(self.params):
            temp0 = fxpMul(self.b1, self.m[idx]) # b1 * m
            temp1 = fxpSub(fxpOne, self.b1) # 1 - b1
            temp1 = fxpMul(temp1, param.grad) # (1 - b1) * g
            self.m[idx] = fxpAdd(temp0, temp1)
            temp0 = fxpMul(self.b2, self.n[idx]) # b2 * n
            temp1 = fxpSub(fxpOne, self.b2) # 1 - b2
            temp1 = fxpMul(temp1, param.grad) # (1 - b2) * g
            temp1 = fxpMul(temp1, param.grad) # (1 - b2) * (g ** 2)
            self.n[idx] = fxpAdd(temp0, temp1)
            temp0 = fxpMul(-at, self.m[idx]) # -at * m
            temp1 = fxpAdd(self.n[idx], self.e) # n + e
            if AdamOptimizer.useLUT:
                # recordLUTUsageStats(temp1, 'rsqrt')
                # temp1 = rsqrtLUT[temp1] # 1 / sqrt(n + e)
                temp1 = fxpSimLUT(temp1, lambda x: x ** -.5)
            else:
                temp1 = AdamOptimizer.fxpRsqrt(temp1)
            deltaParam = fxpMul(temp0, temp1)
            param.update(deltaParam)

# modeled after pytorch's autograd.Function
class Func:
    # forward pass
    # args:
    #   ctx: execution context (empty)
    #   args: tensors only - input tensors
    #   kwargs: non-tensors only - extra computation details e.g., axis of summation
    # returns:
    #   single output tensor - multiple outputs not supported
    @staticmethod
    def fwd(ctx: list, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    # backward pass
    # args:
    #   ctx: execution context (filled in fwd) - if retrieving only 1 object, use a comma e.g., `foo, = ctx`
    #   args: a single output gradient
    # returns:
    #   list or tuple of input gradients
    #     (if returning only 1 input gradient foo, use either `foo,` or `[foo]`)
    @staticmethod
    def bwd(ctx: list, *args) -> list[np.ndarray]:
        raise NotImplementedError

    # do forward pass and record info needed for backward pass
    @classmethod
    def apply(Cls, *args, **kwargs):
        ctx = []
        result = Cls.fwd(ctx, *args, **kwargs)
        result.bwdCls = Cls
        result.bwdCtx = ctx
        result.bwdArgs = args
        return result

# add 2 or more tensors
class AddFunc(Func):
    @staticmethod
    def fwd(ctx, *args: Tensor) -> Tensor:
        assert len(args) >= 2
        resultData = fxpAdd(*[tensor.data for tensor in args])
        ctx += [len(args)]
        result = Tensor(resultData)
        return result

    @staticmethod
    def bwd(ctx, result_grad: np.ndarray) -> list[np.ndarray]:
        numInputs, = ctx
        inputs_grad = (result_grad,) * numInputs
        return inputs_grad

add = AddFunc.apply

# basic subtraction
class SubFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = fxpSub(a.data, b.data)
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        a_grad, b_grad = c_grad, -c_grad
        return a_grad, b_grad

# basic multiply
class MulFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = fxpMul(a.data, b.data)
        ctx += [a.data, b.data]
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        aData, bData = ctx
        a_grad = fxpMul(c_grad, bData)
        b_grad = fxpMul(c_grad, aData)
        return a_grad, b_grad

# array slicing
class GetItemFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, key = None) -> Tensor:
        bData = a.data[key]
        ctx += [a.data.shape, key]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        aShape, key = ctx
        a_grad = np.zeros(aShape, dtype = dtype)
        a_grad[key] = b_grad
        return a_grad,

# sum along one or more axes
class SumFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, axis = None) -> Tensor:
        bData = fxpSum(a.data, axis = axis)
        ctx += [a.data.shape, axis]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        aShape, axis = ctx
        # correctly reshape and expand the output gradient to the input gradient
        if axis:
            temp = np.expand_dims(b_grad, axis)
        else:
            temp = b_grad
        a_grad = np.array(np.broadcast_to(temp, aShape))
        return a_grad,

sum = SumFunc.apply

# basic matrix multiply
class MatrixMultFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = fxpMM(a.data, b.data)
        ctx += [a.data, b.data]
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        aData, bData = ctx
        a_grad = fxpMM(c_grad, bData.T)
        b_grad = fxpMM(aData.T, c_grad)
        return a_grad, b_grad

matmul = MatrixMultFunc.apply

# safe fused multiply add - uses the fxpSFMAdd function
# combines functionality of add, multiply, and mm functions
class SafeFusedMultiplyAddFunc(Func):
    @staticmethod
    def fwd(ctx, *args: Tensor, ops: list = None) -> Tensor:
        argsData = [arg.data for arg in args]
        resultData = fxpSFMAdd(argsData, ops)
        ctx += [argsData, ops]
        result = Tensor(resultData)
        return result

    @staticmethod
    def bwd(ctx, result_grad: np.ndarray) -> list[np.ndarray]:
        argsData, ops = ctx
        inputs_grad = []
        for idx, op in enumerate(ops):
            aData = argsData[idx * 2]
            bData = argsData[idx * 2 + 1]
            if op == np.multiply:
                # derived from MulFunc.bwd
                inputs_grad.append(fxpMul(result_grad, bData))
                inputs_grad.append(fxpMul(result_grad, aData))
            elif op == np.matmul:
                # derived from MatrixMultFunc.bwd
                inputs_grad.append(fxpMM(result_grad, bData.T))
                inputs_grad.append(fxpMM(aData.T, result_grad))
            else:
                raise NotImplemented
        return tuple(inputs_grad)

sfmadd = SafeFusedMultiplyAddFunc.apply

# sigmoid function
class SigmoidFunc(Func):
    useLUT = False
    ppApprox = PiecewisePolynomialEvaluator(
        [fxpMax, 12.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5],
        [-14.0, -10.0, -7.0, -5.5, -4.5, -3.5, -2.75, -2.25, -1.5, -0.75, -0.25],
        [
            [0.0, -3.0517578125e-05, -0.0004730224609375, -0.0020294189453125, -0.00537109375, -0.0135040283203125, -0.024871826171875, -0.0348968505859375, -0.0468902587890625, -0.038848876953125, -0.0152130126953125],
            [0.0, 6.103515625e-05, 0.001007080078125, 0.004150390625, 0.0111236572265625, 0.0290374755859375, 0.0567169189453125, 0.086517333984375, 0.1495208740234375, 0.2174835205078125, 0.2454071044921875],
            [1.0, 0.9999542236328125, 0.99908447265625, 0.9959259033203125, 0.989013671875, 0.9706878662109375, 0.939910888671875, 0.9046478271484375, 0.8175811767578125, 0.6791839599609375, 0.5621795654296875],
        ],
        lambda x: fxpSub(fxpOne, x)
    ).evaluate

    @staticmethod
    def fwd(ctx, a: Tensor) -> Tensor:
        if SigmoidFunc.useLUT:
            # recordLUTUsageStats(a.data, 'sigmoid')
            # bData = sigmoidLUT[a.data]
            bData = fxpSimLUT(a.data, lambda x: 1 / (1 + np.exp(-x)))
            ctx += [a.data]
        else:
            bData = SigmoidFunc.ppApprox(a.data)
            ctx += [bData]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        if SigmoidFunc.useLUT:
            aData, = ctx
            # recordLUTUsageStats(aData, 'dsigmoid')
            # temp0 = dsigmoidLUT[aData]
            temp0 = fxpSimLUT(aData, lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2)
        else:
            bData, = ctx
            # sigmoid'(x) = sigmoid(x) (1 - sigmoid(x))
            temp0 = fxpMul(bData, fxpSub(fxpOne, bData))
        a_grad = fxpMul(b_grad, temp0)
        return a_grad,

sigmoid = SigmoidFunc.apply

# tanh function
class TanhFunc(Func):
    useLUT = False
    ppApprox = PiecewisePolynomialEvaluator(
        [fxpMax, 6.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.125],
        [-7.0, -5.0, -3.5, -2.75, -2.25, -1.875, -1.625, -1.375, -1.125, -0.875, -0.625, -0.375, -0.1875, -0.0625],
        [
            [0.0, -0.000213623046875, -0.003814697265625, -0.0162811279296875, -0.04296875, -0.0858612060546875, -0.13323974609375, -0.1989593505859375, -0.279205322265625, -0.3545684814453125, -0.382720947265625, -0.310760498046875, -0.1786956787109375, -0.06207275390625],
            [0.0, 0.0002593994140625, 0.0040130615234375, 0.0166168212890625, 0.0444793701171875, 0.0902862548828125, 0.144439697265625, 0.226837158203125, 0.3460693359375, 0.505279541015625, 0.6922454833984375, 0.869903564453125, 0.9649810791015625, 0.995330810546875],
            [1.0, 0.999908447265625, 0.9981842041015625, 0.9918670654296875, 0.97802734375, 0.95404052734375, 0.9253387451171875, 0.87982177734375, 0.809295654296875, 0.7039031982421875, 0.554595947265625, 0.3583526611328125, 0.185333251953125, 0.0624237060546875],
        ],
        lambda x: -x
    ).evaluate

    @staticmethod
    def fwd(ctx, a: Tensor) -> Tensor:
        if TanhFunc.useLUT:
            # recordLUTUsageStats(a.data, 'tanh')
            # bData = tanhLUT[a.data]
            bData = fxpSimLUT(a.data, np.tanh)
            ctx += [a.data]
        else:
            bData = TanhFunc.ppApprox(a.data)
            ctx += [bData]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        if TanhFunc.useLUT:
            aData, = ctx
            # recordLUTUsageStats(aData, 'dtanh')
            # temp0 = dtanhLUT[aData]
            temp0 = fxpSimLUT(aData, lambda x: np.cosh(x) ** -2)
        else:
            bData, = ctx
            # tanh'(x) = (1 + tanh(x)) (1 - tanh(x))
            temp0 = fxpMul(fxpAdd(fxpOne, bData), fxpSub(fxpOne, bData))
        a_grad = fxpMul(b_grad, temp0)
        return a_grad,

tanh = TanhFunc.apply

# mean squared error loss
class MeanSquaredErrorLossFunc(Func):
    # assumes the target is in [0,1] and linearly scales the target to [-targetAmp,targetAmp]
    # increasing the static value difference in this manner speeds up training
    # however it also increases likelihood of overflows
    # do not recommend increasing beyond half of max representable value
    targetAmp = 4.

    @staticmethod
    def fwd(ctx, y: Tensor, target: np.ndarray = None, reduction = 'mean') -> Tensor:
        assert target is not None
        assert y.data.ndim == 2
        assert y.data.shape == target.shape
        assert reduction == 'mean' or reduction == 'sum'
        assert np.allclose(np.sum(target, axis = 0), 1)
        # reduction factor for mean or sum
        if reduction == 'mean':
            reductionFactor = 1. / y.data.size
        else:
            reductionFactor = 1. / y.data.shape[0]
        # calculate loss in float - could be made into fixed point
        loss = (fxpToFloats(y.data) + MeanSquaredErrorLossFunc.targetAmp * (1 - 2 * target)) ** 2 / 2
        reducedLoss = np.sum(loss) * reductionFactor
        reducedLoss = np.clip(reducedLoss, -fxpMax, fxpMax)
        ctx += [target, reductionFactor, y.data]
        return Tensor(reducedLoss)

    @staticmethod
    def bwd(ctx, reducedLoss_grad: np.ndarray) -> list[np.ndarray]:
        target, reductionFactor, yData = ctx
        # gradient of loss is calculated in fixedpoint as it is used by backward pass
        target = fxpFromFloats(target)
        targetAmp = fxpFromFloats(MeanSquaredErrorLossFunc.targetAmp)
        reductionFactor = fxpFromFloats(reductionFactor)
        temp0 = fxpAdd(target, target) # 2 * target
        temp0 = fxpSub(fxpOne, temp0) # 1 - 2 * target
        temp0 = fxpMul(targetAmp, temp0) # targetAmp * (1 - 2 * target)
        temp0 = fxpMul(temp0, reductionFactor) # targetAmp * (1 - 2 * target) / reductionFactor
        temp1 = fxpMul(yData, reductionFactor) # y / reductionFactor
        differential = fxpAdd(temp0, temp1)
        y_grad = fxpMul(reducedLoss_grad, differential)
        return y_grad,

meanSquaredErrorLoss = MeanSquaredErrorLossFunc.apply

# one-hot accuracy calculated using floats
def accuracy(y: Tensor, target: np.ndarray):
    argMaxY = np.argmax(y.data, axis = 0)
    hits = target[argMaxY, np.arange(target.shape[1])]
    acc = np.sum(hits) / target.shape[1]
    return acc

