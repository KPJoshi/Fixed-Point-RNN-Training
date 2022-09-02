# this library is extremely similar in structure to `fxpTensor.py` - refer to it for more info

from __future__ import annotations
import builtins
import numpy as np

dtype = np.float32

# store the max. absolute value of each tensor - useful for determining ideal no. of frac bits in fixed point
varBounds = {}
def updateVarBound(var: Tensor, name: str = None) -> None:
    if name is None:
        name = var.name
    currentBound = varBounds.get(name, 0.)
    varBounds[name] = max(currentBound, np.max(np.abs(var.data)))

def printVarBounds() -> None:
    for loc, bound in varBounds.items():
        if bound < 1:
            intBits = 0
        else:
            intBits = int(np.floor(np.log2(bound))) + 1
        print(loc, intBits, sep = '\t')

class Tensor:
    tempCounter = 0

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        if data.dtype == dtype:
            self.data = data
        else:
            self.data = data.astype(dtype)
        self.grad: np.ndarray = None
        self.bwdCls: Func = None
        self.bwdCtx: list = None
        self.bwdArgs: list[Tensor] = None
        if name is not None:
            self.name = name
            updateVarBound(self)
        else:
            self.name = 'tt_' + str(Tensor.tempCounter)
            Tensor.tempCounter += 1

    # topological sort of computation DAG starting from this node
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
        # reverse to get proper order
        gradCalcOrder.reverse()
        # zero all gradients
        for node in gradCalcOrder:
            node.grad = np.zeros_like(node.data)
        # gradient of root is 1
        self.grad = np.ones_like(self.data)
        # backward pass in topological order
        for node in gradCalcOrder:
            if node.bwdArgs:
                inputs_grad = node.bwdCls.bwd(node.bwdCtx, node.grad)
                for input, input_grad in zip(node.bwdArgs, inputs_grad):
                    # print(node.name, '->', input.name, ';', file = sys.stderr)
                    input.grad += input_grad

    def update(self, delta: np.ndarray) -> None:
        self.data += delta
        self.grad = self.bwdCls = self.bwdCtx = self.bwdArgs = None
        updateVarBound(self)

    def __str__(self) -> str:
        return str(self.data)

    def __add__(self, othr: Tensor) -> Tensor:
        return AddFunc.apply(self, othr)

    def __sub__(self, othr: Tensor) -> Tensor:
        return SubFunc.apply(self, othr)

    def __mul__(self, othr: Tensor) -> Tensor:
        return MulFunc.apply(self, othr)

    def __getitem__(self, key) -> Tensor:
        return GetItemFunc.apply(self, key = key)

class AdamOptimizer:
    def __init__(self, params: list[Tensor], a = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-8):
        self.params = params
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.b1t = 1.
        self.b2t = 1.
        self.m = []
        self.n = []
        for param in params:
            self.m.append(np.zeros_like(param.data))
            self.n.append(np.zeros_like(param.data))

    def step(self) -> None:
        # update hyperparams
        self.b1t *= self.b1
        self.b2t *= self.b2
        at = self.a * np.sqrt(1 - self.b2t) / (1 - self.b1t)
        # update moments and param
        for idx, param in enumerate(self.params):
            self.m[idx] = self.b1 * self.m[idx] + (1 - self.b1) * param.grad
            self.n[idx] = self.b2 * self.n[idx] + (1 - self.b2) * (param.grad ** 2)
            temp = self.m[idx] / np.sqrt(self.n[idx] + self.e)
            # temp = self.m[idx] / (np.sqrt(self.n[idx]) + self.e)
            # temp = self.m[idx] / np.sqrt(self.n[idx])
            # temp = self.m[idx] / np.sqrt(self.n[idx] + self.e ** 2)
            deltaParam = -at * temp
            param.update(deltaParam)

class Func:
    @staticmethod
    def fwd(ctx: list, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def bwd(ctx: list, *args) -> list[np.ndarray]:
        raise NotImplementedError

    @classmethod
    def apply(Cls, *args, **kwargs):
        ctx = []
        result = Cls.fwd(ctx, *args, **kwargs)
        result.bwdCls = Cls
        result.bwdCtx = ctx
        result.bwdArgs = args
        return result

class AddFunc(Func):
    @staticmethod
    def fwd(ctx, *args: Tensor) -> Tensor:
        assert len(args) >= 2
        resultData = builtins.sum(tensor.data for tensor in args)
        ctx += [len(args)]
        result = Tensor(resultData)
        return result

    @staticmethod
    def bwd(ctx, result_grad: np.ndarray) -> list[np.ndarray]:
        numInputs, = ctx
        inputs_grad = (result_grad,) * numInputs
        return inputs_grad

add = AddFunc.apply

class SubFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = a.data - b.data
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        a_grad, b_grad = c_grad, -c_grad
        return a_grad, b_grad

class MulFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = a.data * b.data
        ctx += [a.data, b.data]
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        aData, bData = ctx
        a_grad = c_grad * bData
        b_grad = c_grad * aData
        return a_grad, b_grad

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

class SumFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, axis = None) -> Tensor:
        bData = np.sum(a.data, axis = axis)
        ctx += [a.data.shape, axis]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        aShape, axis = ctx
        if axis:
            temp = np.expand_dims(b_grad, axis)
        else:
            temp = b_grad
        a_grad = np.array(np.broadcast_to(temp, aShape))
        return a_grad,

sum = SumFunc.apply

class MatrixMultFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor, b: Tensor) -> Tensor:
        cData = np.matmul(a.data, b.data)
        ctx += [a.data, b.data]
        c = Tensor(cData)
        return c

    @staticmethod
    def bwd(ctx, c_grad: np.ndarray) -> list[np.ndarray]:
        aData, bData = ctx
        a_grad = np.matmul(c_grad, bData.T)
        b_grad = np.matmul(aData.T, c_grad)
        return a_grad, b_grad

matmul = MatrixMultFunc.apply

class SafeFusedMultiplyAddFunc(Func):
    @staticmethod
    def fwd(ctx, *args: Tensor, ops: list = None) -> Tensor:
        assert len(args) == len(ops) * 2
        argsData = [arg.data for arg in args]
        resultData = np.zeros((argsData[0].shape[0], argsData[1].shape[1]), dtype = dtype)
        for idx, op in enumerate(ops):
            resultData += op(argsData[idx * 2], argsData[idx * 2 + 1])
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
                inputs_grad.append(result_grad * bData)
                inputs_grad.append(result_grad * aData)
            elif op == np.matmul:
                # derived from MatrixMultFunc.bwd
                inputs_grad.append(np.matmul(result_grad, bData.T))
                inputs_grad.append(np.matmul(aData.T, result_grad))
            else:
                raise NotImplemented
        return tuple(inputs_grad)

sfmadd = SafeFusedMultiplyAddFunc.apply

class SigmoidFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor) -> Tensor:
        bData = 1 / (1 + np.exp(-a.data))
        ctx += [bData]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        bData, = ctx
        a_grad = b_grad * (bData * (1 - bData))
        return a_grad,

sigmoid = SigmoidFunc.apply

class TanhFunc(Func):
    @staticmethod
    def fwd(ctx, a: Tensor) -> Tensor:
        bData = np.tanh(a.data)
        ctx += [bData]
        b = Tensor(bData)
        return b

    @staticmethod
    def bwd(ctx, b_grad: np.ndarray) -> list[np.ndarray]:
        bData, = ctx
        a_grad = b_grad * ((1 + bData) * (1 - bData))
        return a_grad,

tanh = TanhFunc.apply

class SoftMaxCrossEntropyLossFunc(Func):
    @staticmethod
    def fwd(ctx, y: Tensor, target: np.ndarray = None, reduction = 'mean') -> Tensor:
        assert target is not None
        assert y.data.ndim == 2
        assert y.data.shape == target.shape
        assert reduction == 'mean' or reduction == 'sum'
        assert np.allclose(np.sum(target, axis = 0), 1)
        # reduction factor for mean or sum
        if reduction == 'mean':
            reductionFactor = 1. / y.data.shape[1]
        else:
            reductionFactor = 1.
        # log-sum-exp trick
        yShift = np.max(y.data, axis = 0)
        # if shift is negative, don't shift
        yShift[yShift < 0] = 0
        yShifted = y.data - yShift
        expY = np.exp(yShifted)
        sumExpY = np.sum(expY, axis = 0)
        logSumExpY = np.log(sumExpY)
        loss = logSumExpY - np.sum(y.data * target, axis = 0)
        reducedLoss = np.sum(loss) * reductionFactor
        softMaxY = expY / sumExpY
        softMaxYMinusTarget = softMaxY - target
        ctx += [reductionFactor, softMaxYMinusTarget]
        return Tensor(reducedLoss)

    @staticmethod
    def bwd(ctx, reducedLoss_grad: np.ndarray) -> list[np.ndarray]:
        reductionFactor, softMaxYMinusTarget = ctx
        temp0 = reducedLoss_grad * reductionFactor
        loss_grad = np.full_like(softMaxYMinusTarget, temp0)
        y_grad = loss_grad * softMaxYMinusTarget
        return y_grad,

softmaxCrossEntropyLoss = SoftMaxCrossEntropyLossFunc.apply

class MeanSquaredErrorLossFunc(Func):
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
        loss = (y.data + MeanSquaredErrorLossFunc.targetAmp * (1 - 2 * target)) ** 2 / 2
        reducedLoss = np.sum(loss) * reductionFactor
        ctx += [target, reductionFactor, y.data]
        return Tensor(reducedLoss)

    @staticmethod
    def bwd(ctx, reducedLoss_grad: np.ndarray) -> list[np.ndarray]:
        target, reductionFactor, yData = ctx
        differential = MeanSquaredErrorLossFunc.targetAmp * (1 - 2 * target) + yData
        differential *= reductionFactor
        y_grad = reducedLoss_grad * differential
        return y_grad,

meanSquaredErrorLoss = MeanSquaredErrorLossFunc.apply

def accuracy(y: Tensor, target: np.ndarray):
    argMaxY = np.argmax(y.data, axis = 0)
    hits = target[argMaxY, np.arange(target.shape[1])]
    acc = np.sum(hits) / target.shape[1]
    return acc
