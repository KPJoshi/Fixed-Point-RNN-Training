# a pytorch autograd function that
# 1) converts float input to fixpoint
# 2) calculates sigmoid or tanh in fixed point using simulated LUTs
# 3) converts result back to float and returns it
# 4) also calculates gradients via fixed point

import torch

intBits = 2
fracBits = 5
lowerClip = -(2 ** intBits)
upperClip = -lowerClip - (2 ** -fracBits)

def roundAndClip(T: torch.tensor) -> torch.tensor:
    roundedT = torch.round(T * (2 ** fracBits)) / (2 ** fracBits)
    clippedT = torch.clip(roundedT, lowerClip, upperClip)
    return clippedT

class LUTSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        adjInput = roundAndClip(input)
        output = torch.sigmoid(adjInput)
        ctx.save_for_backward(output)
        adjOutput = roundAndClip(output)
        return adjOutput

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[0]:
            output = ctx.saved_tensors[0]
            derivative = output * (1 - output)
            adjDerivative = roundAndClip(derivative)
            grad_input = grad_output * adjDerivative

        return grad_input

class LUTTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        adjInput = roundAndClip(input)
        output = torch.tanh(adjInput)
        ctx.save_for_backward(output)
        adjOutput = roundAndClip(output)
        return adjOutput

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[0]:
            output = ctx.saved_tensors[0]
            derivative = (1 + output) * (1 - output)
            adjDerivative = roundAndClip(derivative)
            grad_input = grad_output * adjDerivative

        return grad_input
