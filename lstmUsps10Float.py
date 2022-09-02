#! /usr/bin/env python3

# this is extremely similar to `lstmUsps10.py` - refer to it for more info
# also records the dynamic range of each tensor for reference when deciding the shift

from __future__ import annotations
from datetime import datetime
import numpy as np
import os
import time

import myTensor

# input info
inputDir = 'data/usps10/'
numClasses = 10
numFeats = 16
timeSteps = 16
# training params
rngSeed = 0
numEpochs = 300
hiddenDim = 10
batchSize = 128
rankW = 4
rankU = 0

# data preprocessing function
def processData(filename: str) -> tuple[np.ndarray, np.ndarray]:
    rawData = np.load(filename)
    # rawData[:, 0] is labels
    # rawData[:, 1:] is normalized features of length 256
    # initial extraction
    labels = np.array(rawData[:, 0], dtype = int)
    features = rawData[:, 1:]
    # process labels
    labels -= np.min(labels)
    assert numClasses == np.max(labels) + 1
    denseLabels = np.zeros((numClasses, labels.shape[0]), dtype = int)
    denseLabels[labels, np.arange(labels.shape[0])] = 1
    labels = denseLabels
    # process features
    assert features.shape[1] == numFeats * timeSteps
    features = np.transpose(features.reshape((-1, timeSteps, numFeats)), (1, 2, 0))
    return features, labels

# get processed data
trainX, trainY = processData(inputDir + 'train.npy')
testX, testY = processData(inputDir + 'test.npy')
trainX = myTensor.Tensor(trainX, 'trainX')
testX = myTensor.Tensor(testX, 'testX')

# generate random params
rng = np.random.default_rng(rngSeed)
# input weights
if rankW:
    Wr = myTensor.Tensor(rng.normal(0, 0.1, (rankW, numFeats)), 'Wr')
    inDimW = rankW
else:
    inDimW = numFeats
Wf = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wf')
Wi = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wi')
Wc = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wc')
Wo = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wo')
# hidden state weights
if rankU:
    Ur = myTensor.Tensor(rng.normal(0, 0.1, (rankU, hiddenDim)), 'Ur')
    inDimU = rankU
else:
    inDimU = hiddenDim
Uf = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uf')
Ui = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Ui')
Uc = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uc')
Uo = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uo')
# biases
bf = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bf')
bi = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bi')
bc = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bc')
bo = myTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bo')
# final fully-connected layer params
Wl = myTensor.Tensor(rng.normal(0, 0.1, (numClasses, hiddenDim)), 'Wl')
bl = myTensor.Tensor(rng.normal(0, 0.1, (numClasses, 1)), 'bl')
# list of params to update
params = [Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo, Wl, bl]
if rankW:
    params.append(Wr)
if rankU:
    params.append(Ur)

def forward(X: myTensor.Tensor) -> myTensor.Tensor:
    # utility data
    thisBatchSize = X.data.shape[2]
    bExpander = myTensor.Tensor(np.ones((1, thisBatchSize), dtype = float), 'bX')
    # initialize hidden and cell state
    H = myTensor.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'H0')
    C = myTensor.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'C0')
    # process each time step
    for timeStep in range(timeSteps):
        # extract current timestep input
        Xt = X[timeStep]
        Xt.name = 'Xt' + str(timeStep)
        if rankW:
            Xt = myTensor.matmul(Wr, Xt)
            myTensor.updateVarBound(Xt, 'Wr@Xt')
        if rankU:
            H = myTensor.matmul(Ur, H)
            myTensor.updateVarBound(H, 'Ur@H')
        # LSTM ops
        GfRaw = myTensor.sfmadd(Wf, Xt, Uf, H, bf, bExpander, ops = [np.matmul, np.matmul, np.matmul])
        myTensor.updateVarBound(GfRaw, 'GfRaw')
        Gf = myTensor.sigmoid(GfRaw)
        Gf.name = 'Gf' + str(timeStep)
        GiRaw = myTensor.sfmadd(Wi, Xt, Ui, H, bi, bExpander, ops = [np.matmul, np.matmul, np.matmul])
        myTensor.updateVarBound(GiRaw, 'GiRaw')
        Gi = myTensor.sigmoid(GiRaw)
        Gi.name = 'Gi' + str(timeStep)
        GoRaw = myTensor.sfmadd(Wo, Xt, Uo, H, bo, bExpander, ops = [np.matmul, np.matmul, np.matmul])
        myTensor.updateVarBound(GoRaw, 'GoRaw')
        Go = myTensor.sigmoid(GoRaw)
        Go.name = 'Go' + str(timeStep)
        GcRaw = myTensor.sfmadd(Wc, Xt, Uc, H, bc, bExpander, ops = [np.matmul, np.matmul, np.matmul])
        myTensor.updateVarBound(GcRaw, 'GcRaw')
        Gc = myTensor.tanh(GcRaw)
        Gc.name = 'Gc' + str(timeStep)
        newC = myTensor.sfmadd(C, Gf, Gc, Gi, ops = [np.multiply, np.multiply])
        newH = Go * myTensor.tanh(newC)
        # feedback
        H = newH
        H.name = 'H' + str(timeStep + 1)
        C = newC
        C.name = 'C' + str(timeStep + 1)
        myTensor.updateVarBound(C)
    # final layer
    Y_pred = myTensor.sfmadd(Wl, H, bl, bExpander, ops = [np.matmul, np.matmul])
    Y_pred.name = 'Yp'
    myTensor.updateVarBound(Y_pred)
    return Y_pred

optimizer = myTensor.AdamOptimizer(params)
trainingSessionData = np.empty((numEpochs, 6))
numTrainInputs = trainX.data.shape[2]
numBatches = int(np.ceil(numTrainInputs / batchSize))
for epoch in range(numEpochs):
    # permutation for training data
    permutation = rng.permutation(numTrainInputs)
    trainLoss = trainAcc = fwdTime = bwdTime = 0.
    for batchIdx in range(numBatches):
        # extract batch data
        batchInputIdxs = permutation[batchIdx * batchSize : (batchIdx + 1) * batchSize]
        batchX = trainX[:, :, batchInputIdxs]
        batchX.name = 'batchX'
        batchY = trainY[:, batchInputIdxs]
        # forward
        startTime = time.time()
        Y_pred = forward(batchX)
        # backward
        # batchLoss = myTensor.softmaxCrossEntropyLoss(Y_pred, target = batchY)
        batchLoss = myTensor.meanSquaredErrorLoss(Y_pred, target = batchY)
        batchLoss.name = 'batchLoss'
        myTensor.updateVarBound(batchLoss)
        batchAcc = myTensor.accuracy(Y_pred, batchY)
        fwdTime += time.time() - startTime
        startTime = time.time()
        batchLoss.backward()
        # update
        optimizer.step()
        bwdTime += time.time() - startTime
        # store loss and accuracy data
        trainLoss += batchLoss.data * batchInputIdxs.size
        trainAcc += batchAcc * batchInputIdxs.size
    # calculate average training loss and accuracy
    trainLoss /= numTrainInputs
    trainAcc /= numTrainInputs
    # validation
    startTime = time.time()
    Y_pred = forward(testX)
    # testLoss = myTensor.softmaxCrossEntropyLoss(Y_pred, target = testY)
    testLoss = myTensor.meanSquaredErrorLoss(Y_pred, target = testY)
    myTensor.updateVarBound(testLoss, 'testLoss')
    testAcc = myTensor.accuracy(Y_pred, testY)
    fwdTime += time.time() - startTime
    outputFormat = 'Epoch {:4}:\n'\
                   'Train: Loss {:7.4f}\tAccuracy {:6.4f}\n'\
                   'Test : Loss {:7.4f}\tAccuracy {:6.4f}\n'\
                   'Time : Forward {:5.2f}s\tBackward {:5.2f}s'
    trainingSessionData[epoch] = trainLoss, trainAcc, testLoss.data, testAcc, fwdTime, bwdTime
    print(outputFormat.format(epoch, *trainingSessionData[epoch]))

# show variable bounds
myTensor.printVarBounds()

# dump model
os.makedirs('./models/', exist_ok = True)
modelFile = './models/' + datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
paramDict = {param.name : param.data for param in params}
np.savez_compressed('{}.npz'.format(modelFile), **paramDict)
print('Model saved to {}.npz'.format(modelFile))
np.savetxt('{}.csv'.format(modelFile), trainingSessionData, delimiter = ',')
print('Training session data saved to {}.csv'.format(modelFile))
