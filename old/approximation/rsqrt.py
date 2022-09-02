import numpy as np

from piecewisePolyEvalr import PiecewisePolynomialEvaluator

functionName = 'rsqrt'

# 16/12
segmentMaxs = [4, 3, 2, 1.5]
segmentDeltas = [-3.5, -2.5, -1.75, -1.25]
segmentCoeffs = [
    [-0.003950320480075422, -0.012999679184045975, -0.044692774389130226, -0.1470745808686127],
    [0.016574340317167813, 0.038920349593374176, 0.09375862745570451, 0.2201667449890063],
    [-0.07635899286456249, -0.1264823122153712, -0.21597584663642982, -0.3577460026705722],
    [0.5345171897256933, 0.6324311259242547, 0.7559214590316282, 0.8943926755489677],
]

def rsqrt(x: np.ndarray) -> np.ndarray:
    return x ** -.5

nothing = [0, 0] # negative x is invalid for rsqrt

rsqrtPolyEvaluator = PiecewisePolynomialEvaluator(segmentMaxs, segmentDeltas, segmentCoeffs, nothing)

goldenFunc = rsqrt
plynmlFunc = rsqrtPolyEvaluator.floatEval
fixPntFunc = rsqrtPolyEvaluator.fixPtEval
