#! /usr/bin/env python3

import fxpEzPCGen

# input info
numClasses = 'NumClasses' # number of classes
numFeats = 'NumFeatures' # features per time step
timeSteps = 'TimeSteps' # time steps
timeSteps_actual = 16 # needed to actually generate the unrolled code
# training params
hiddenDim = 'HiddenDim' # hidden dimension (cell state / hidden state) of LSTM
batchSize = 'BatchSize' # batch size 
rankW = 'RankW' # rank of W matrices - set to None to use full rank
rankU = None # rank of U matrices - set to None to use full rank

# input data
X = fxpEzPCGen.Tensor((timeSteps, batchSize, numFeats), False, False, 'batchX')
Y = 'batchTarget'

# params
# input matrices
if rankW:
    Wr = fxpEzPCGen.Tensor((numFeats, rankW), True, True, 'Wr')
    inDimW = rankW
else:
    inDimW = numFeats
Wf = fxpEzPCGen.Tensor((inDimW, hiddenDim), True, True, 'Wf')
Wi = fxpEzPCGen.Tensor((inDimW, hiddenDim), True, True, 'Wi')
Wc = fxpEzPCGen.Tensor((inDimW, hiddenDim), True, True, 'Wc')
Wo = fxpEzPCGen.Tensor((inDimW, hiddenDim), True, True, 'Wo')
# hidden state matrices
if rankU:
    Ur = fxpEzPCGen.Tensor((hiddenDim, rankU), True, True, 'Ur')
    inDimU = rankU
else:
    inDimU = hiddenDim
Uf = fxpEzPCGen.Tensor((inDimU, hiddenDim), True, True, 'Uf')
Ui = fxpEzPCGen.Tensor((inDimU, hiddenDim), True, True, 'Ui')
Uc = fxpEzPCGen.Tensor((inDimU, hiddenDim), True, True, 'Uc')
Uo = fxpEzPCGen.Tensor((inDimU, hiddenDim), True, True, 'Uo')
# biases
Bf = fxpEzPCGen.Tensor((hiddenDim,), True, True, 'Bf')
Bi = fxpEzPCGen.Tensor((hiddenDim,), True, True, 'Bi')
Bc = fxpEzPCGen.Tensor((hiddenDim,), True, True, 'Bc')
Bo = fxpEzPCGen.Tensor((hiddenDim,), True, True, 'Bo')
# output layer
Wl = fxpEzPCGen.Tensor((hiddenDim, numClasses), True, True, 'Wl')
Bl = fxpEzPCGen.Tensor((numClasses,), True, True, 'Bl')

# forward pass
H = fxpEzPCGen.zeros((batchSize, hiddenDim))
C = fxpEzPCGen.zeros((batchSize, hiddenDim))
for timeStep in range(timeSteps_actual):
    Xt = fxpEzPCGen.getTimestep(X, timeStep)
    if rankW:
        Xt = fxpEzPCGen.matmul(Xt, Wr)
    if rankU:
        H = fxpEzPCGen.matmul(H, Ur)
    Gf = fxpEzPCGen.sigmoid(fxpEzPCGen.matmul(Xt, Wf) + fxpEzPCGen.matmul(H, Uf) + Bf)
    Gi = fxpEzPCGen.sigmoid(fxpEzPCGen.matmul(Xt, Wi) + fxpEzPCGen.matmul(H, Ui) + Bi)
    Gc = fxpEzPCGen.tanh   (fxpEzPCGen.matmul(Xt, Wc) + fxpEzPCGen.matmul(H, Uc) + Bc)
    Go = fxpEzPCGen.sigmoid(fxpEzPCGen.matmul(Xt, Wo) + fxpEzPCGen.matmul(H, Uo) + Bo)
    C = (C * Gf) + (Gc * Gi)
    H = Go * fxpEzPCGen.tanh(C)
Pred = fxpEzPCGen.matmul(H, Wl) + Bl
fxpEzPCGen.meanSquaredErrorLoss(Pred, Y)

fxpEzPCGen.writeInit('autogen/init.ezpc')
fxpEzPCGen.writeTest('autogen/testFunc.ezpc', 'autogen/testCall.ezpc')
fxpEzPCGen.writeTrain('autogen/trainFunc.ezpc', 'autogen/trainCall.ezpc')
