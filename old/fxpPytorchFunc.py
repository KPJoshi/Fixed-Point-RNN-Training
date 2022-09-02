#! /usr/bin/env python3

# a pytorch autograd function that
# 1) converts float input to fixpoint
# 2) calculates sigmoid or tanh in fixed point using polynomial approx
# 3) converts result back to float and returns it
# 4) also calculates gradients via fixed point

import numpy as np
import torch

from fixpoint import FixedPoint

dType = np.int32
shift = 24

def fixPtEval(fpX: FixedPoint, segmentMaxs, segmentDeltas, segmentCoeffs, negativeHandling) -> FixedPoint:
    fp0 = FixedPoint.fromFloats(0, dType, shift) # 0
    negatives = fpX.compare(fp0) < 0
    fpAbs = FixedPoint.fromFloats(np.choose(negatives, [1,-1]), dType, shift) # for making all x positive
    fpX *= fpAbs
    segmentIdx = np.zeros(negatives.shape, dtype=int) # assume first segment up to inf
    for idx, segmentMax in enumerate(segmentMaxs):
        leqMax = fpX.compare(FixedPoint.fromFloats(segmentMax, dType, shift)) <= 0
        segmentIdx[leqMax] = idx
    fpDeltas = segmentDeltas.choose(segmentIdx)
    fpX += fpDeltas
    fpAccumulator = FixedPoint.fromFloats(np.zeros(fpX.shiftedValues.shape), dType, shift)
    for coeff in segmentCoeffs:
        fpCoeff = coeff.choose(segmentIdx)
        fpAccumulator = (fpAccumulator * fpX) + fpCoeff
    fpNegativeHandling0 = FixedPoint.fromFloats(np.choose(negatives, [0,negativeHandling[0]]), dType, shift)
    fpNegativeHandling1 = FixedPoint.fromFloats(np.choose(negatives, [1,negativeHandling[1]]), dType, shift)
    fpAccumulator = fpNegativeHandling0 + fpAccumulator * fpNegativeHandling1
    return fpAccumulator

class FxpSigmoid(torch.autograd.Function):

    segmentMaxs = [np.inf, 9, 6, 4, 2, 1]
    segmentDeltas = [-9.0, -7.5, -5.0, -3.0, -1.5, -0.5]
    segmentCoeffs = [
        [0.0, -2.6164424960378507e-05, -0.0002633862733478272, -0.0007083833560752303, 0.003144512209037246, 0.004182553722671427],
        [0.0, 0.0001035454117896343, 0.0011135130881650434, 0.005425730282855462, 0.002340903203909948, -0.015748652225848862],
        [0.0, -0.00027229431628598164, -0.0032733128970798948, -0.020466594620509353, -0.04737148558236161, -0.02876298587976544],
        [0.0, 0.0005467994467164436, 0.006637464789730501, 0.0451914154687907, 0.14916091568948991, 0.23498706183016754],
        [0.999983298578152, 0.9994466178651105, 0.9933066601574142, 0.9525748333247185, 0.8175746409205796, 0.6224591405275942],
    ]
    segments = len(segmentMaxs)
    negativeHandling = [1, -1] # for negative x, subtract from 1
    segmentDeltas = FixedPoint.fromFloats(np.array(segmentDeltas), dType, shift)
    for idx in range(len(segmentCoeffs)):
        segmentCoeffs[idx] = FixedPoint.fromFloats(np.array(segmentCoeffs[idx]), dType, shift)

    @staticmethod
    def forward(ctx, input):
        fxp_input = FixedPoint.fromFloats(input.numpy(), dType, shift)
        fxp_output = fixPtEval(fxp_input, FxpSigmoid.segmentMaxs, FxpSigmoid.segmentDeltas, FxpSigmoid.segmentCoeffs, FxpSigmoid.negativeHandling)
        output = torch.from_numpy(fxp_output.toFloats())
        ctx.fxp_output = fxp_output # for fixpoint gradient calc
        ctx.save_for_backward(output) # for actual gradient calc
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[0]:
            fxp_output = ctx.fxp_output
            fxp_one = FixedPoint.fromFloats(1, dType, shift)
            fxp_grad_output = FixedPoint.fromFloats(grad_output.numpy(), dType, shift)
            fxp_grad_input = fxp_grad_output * fxp_output * (fxp_one - fxp_output)
            grad_input = torch.from_numpy(fxp_grad_input.toFloats())
            output = ctx.saved_tensors[0]
            actual_grad_input = grad_output * output * (1 - output)
            print(torch.sqrt((grad_input - actual_grad_input).pow(2).mean()))

        return grad_input

class FxpTanh(torch.autograd.Function):

    segmentMaxs = [np.inf, 9, 6, 4, 2, 1]
    segmentDeltas = [-9.0, -7.5, -5.0, -3.0, -1.5, -0.5]
    segmentCoeffs = [
        [0.0, -6.298642302836332e-07, -7.35632838178498e-05, -0.0037725982505670544, -0.022805526501813903, 0.15176118137775124],
        [0.0, 1.3133484900104593e-06, 0.00015034792795265162, 0.007922086552004829, 0.08682045226647597, -0.10004467293661844],
        [0.0, -1.0215520870668847e-06, -0.0001764687965768404, -0.009578659814979514, -0.16370326978011165, -0.3628375850381829],
        [0.0, 9.712435902778152e-07, 0.00017514925988192142, 0.00955199604883907, 0.18076434489087528, 0.7867306372114518],
        [0.9999999994421064, 0.9999993609137693, 0.9999089044988944, 0.9950401441910622, 0.9051489253499478, 0.4621202860587143],
    ]
    segments = len(segmentMaxs)
    negativeHandling = [0, -1] # for negative x, subtract from 0
    segmentDeltas = FixedPoint.fromFloats(np.array(segmentDeltas), dType, shift)
    for idx in range(len(segmentCoeffs)):
        segmentCoeffs[idx] = FixedPoint.fromFloats(np.array(segmentCoeffs[idx]), dType, shift)

    @staticmethod
    def forward(ctx, input):
        fxp_input = FixedPoint.fromFloats(input.numpy(), dType, shift)
        fxp_output = fixPtEval(fxp_input, FxpTanh.segmentMaxs, FxpTanh.segmentDeltas, FxpTanh.segmentCoeffs, FxpTanh.negativeHandling)
        output = torch.from_numpy(fxp_output.toFloats())
        ctx.fxp_output = fxp_output # for fixpoint gradient calc
        ctx.save_for_backward(output) # for actual gradient calc
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[0]:
            fxp_output = ctx.fxp_output
            fxp_one = FixedPoint.fromFloats(1, dType, shift)
            fxp_grad_output = FixedPoint.fromFloats(grad_output.numpy(), dType, shift)
            fxp_grad_input = fxp_grad_output * (fxp_one + fxp_output) * (fxp_one - fxp_output)
            grad_input = torch.from_numpy(fxp_grad_input.toFloats())
            output = ctx.saved_tensors[0]
            actual_grad_input = grad_output * (1 + output) * (1 - output)
            print(torch.sqrt((grad_input - actual_grad_input).pow(2).mean()))

        return grad_input

x = torch.linspace(-25, 25, 50001, device=torch.device('cpu'), dtype=torch.float, requires_grad=True)

# y = torch.sigmoid(x)
# y_pred = FxpSigmoid.apply(x)
y = torch.tanh(x)
y_pred = FxpTanh.apply(x)

print(torch.sqrt((y_pred - y).pow(2).mean()))

loss = (y_pred - y).pow(2).sum()

loss.backward()
