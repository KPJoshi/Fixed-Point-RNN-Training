import numpy as np

from piecewisePolyEvalr import PiecewisePolynomialEvaluator

functionName = 'sigmoid'

# old
segmentMaxs = [8, 6, 4, 3, 2, 1.5, 1, 0.5]
segmentDeltas = [-7.0, -5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25]
segmentCoeffs = [
    [0.00015730430369424375, 0.0011022663154184628, 0.003944662485889925, 0.006704427399559228, 0.005063804901228594, -0.001160335070084228, -0.011146653048269001, -0.019462722719731224],
    [-0.00048715847286818353, -0.0034987079649739794, -0.013549346819122579, -0.029813410486695877, -0.044289500062103804, -0.047771557925592734, -0.03876207444967916, -0.015171146380156264],
    [0.0009095768698408153, 0.006644208888040318, 0.02845272323903174, 0.07010534380044162, 0.1261295322911921, 0.1731050734100938, 0.21789489526382733, 0.24613347735196056],
    [0.9990922468768384, 0.9933291140792209, 0.9706916885407273, 0.924143791448448, 0.8519521640401329, 0.7772984217199518, 0.6791769498899906, 0.5621756727861847],
]
# <= 3 ULPs
segmentMaxs = [8, 6, 4, 3, 2, 1, 0.5]
segmentDeltas = [-7.0, -5.0, -3.5, -2.5, -1.5, -0.75, -0.25]
segmentCoeffs = [
    [-0.0004771806822019899, -0.0034324356584245616, -0.013502384239061565, -0.02979025867908921, -0.04689776255866341, -0.03884589545740016, -0.015210804630263888],
    [0.001003437333646023, 0.007302479571585665, 0.029044187011066244, 0.07111232106490548, 0.1495161853661976, 0.21747681647894404, 0.2454031425902522],
    [0.9990889488055992, 0.993307149075715, 0.9706877692486433, 0.9241418199787562, 0.8175744761936433, 0.6791786991753929, 0.562176500885798],
]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

subtractFromOne = [1, -1] # for handling negative x

sigmoidPolyEvaluator = PiecewisePolynomialEvaluator(segmentMaxs, segmentDeltas, segmentCoeffs, subtractFromOne)

goldenFunc = sigmoid
plynmlFunc = sigmoidPolyEvaluator.floatEval
fixPntFunc = sigmoidPolyEvaluator.fixPtEval