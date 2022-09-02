#! /usr/bin/env python3

from __future__ import annotations
from datetime import datetime
import numpy as np
import os
import time

import fxpTensor

# implements a (optionally low-rank) LSTM cell and trains it on the USPS 10 dataset

# input info
inputDir = 'data/usps10/' # location of train and test data
numClasses = 10 # number of classes
numFeats = 16 # features per time step
timeSteps = 16 # time steps
# training params
rngSeed = 0 # seed for initializing params, shuffling training data
numEpochs = 300
hiddenDim = 10 # hidden dimension (cell state / hidden state) of LSTM
batchSize = 128
rankW = 4 # rank of W matrices - set to 0 to use full rank
rankU = 0 # rank of U matrices - set to 0 to use full rank

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
    # transform labels to target matrix (0: wrong label, 1: correct label)
    denseLabels = np.zeros((numClasses, labels.shape[0]), dtype = int)
    denseLabels[labels, np.arange(labels.shape[0])] = 1
    labels = denseLabels
    # process features
    assert features.shape[1] == numFeats * timeSteps
    # rearrange data - dimensions become [sampleIdx, timestep, feature]
    features = np.transpose(features.reshape((-1, timeSteps, numFeats)), (1, 2, 0))
    return features, labels

# get processed data
trainX, trainY = processData(inputDir + 'train.npy')
testX, testY = processData(inputDir + 'test.npy')
trainX = fxpTensor.Tensor(trainX, 'trainX')
testX = fxpTensor.Tensor(testX, 'testX')

# generate random params
rng = np.random.default_rng(rngSeed)
# input weights
if rankW:
    Wr = fxpTensor.Tensor(rng.normal(0, 0.1, (rankW, numFeats)), 'Wr')
    inDimW = rankW
else:
    inDimW = numFeats
Wf = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wf')
Wi = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wi')
Wc = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wc')
Wo = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimW)), 'Wo')
# hidden state weights
if rankU:
    Ur = fxpTensor.Tensor(rng.normal(0, 0.1, (rankU, hiddenDim)), 'Ur')
    inDimU = rankU
else:
    inDimU = hiddenDim
Uf = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uf')
Ui = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Ui')
Uc = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uc')
Uo = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, inDimU)), 'Uo')
# biases
bf = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bf')
bi = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bi')
bc = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bc')
bo = fxpTensor.Tensor(rng.normal(0, 0.1, (hiddenDim, 1)), 'bo')
# final fully-connected layer params
Wl = fxpTensor.Tensor(rng.normal(0, 0.1, (numClasses, hiddenDim)), 'Wl')
bl = fxpTensor.Tensor(rng.normal(0, 0.1, (numClasses, 1)), 'bl')
# list of params for ADAM to update
params = [Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo, Wl, bl]
if rankW:
    params.append(Wr)
if rankU:
    params.append(Ur)

def forward(X: fxpTensor.Tensor) -> fxpTensor.Tensor:
    # utility data
    thisBatchSize = X.data.shape[2]
    bExpander = fxpTensor.Tensor(np.ones((1, thisBatchSize), dtype = float), 'bX') # hack for broadcasting the bias matrix
    # initialize hidden and cell state
    H = fxpTensor.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'H0')
    C = fxpTensor.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'C0')
    # process each time step
    for timeStep in range(timeSteps):
        # extract current timestep input
        Xt = X[timeStep]
        Xt.name = 'Xt' + str(timeStep)
        if rankW:
            Xt = fxpTensor.matmul(Wr, Xt)
        if rankU:
            H = fxpTensor.matmul(Ur, H)
        # LSTM ops
        Gf = fxpTensor.sigmoid(fxpTensor.sfmadd(Wf, Xt, Uf, H, bf, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Gf.name = 'Gf' + str(timeStep)
        Gi = fxpTensor.sigmoid(fxpTensor.sfmadd(Wi, Xt, Ui, H, bi, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Gi.name = 'Gi' + str(timeStep)
        Go = fxpTensor.sigmoid(fxpTensor.sfmadd(Wo, Xt, Uo, H, bo, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Go.name = 'Go' + str(timeStep)
        Gc = fxpTensor.tanh(fxpTensor.sfmadd(Wc, Xt, Uc, H, bc, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Gc.name = 'Gc' + str(timeStep)
        newC = fxpTensor.sfmadd(C, Gf, Gc, Gi, ops = [np.multiply, np.multiply])
        newH = Go * fxpTensor.tanh(newC)
        # feedback
        H = newH
        H.name = 'H' + str(timeStep + 1)
        C = newC
        C.name = 'C' + str(timeStep + 1)
    # final layer
    Y_pred = fxpTensor.sfmadd(Wl, H, bl, bExpander, ops = [np.matmul, np.matmul])
    Y_pred.name = 'Yp'
    return Y_pred

optimizer = fxpTensor.AdamOptimizer(params) # use ADAM optimizer
trainingSessionData = np.empty((numEpochs, 6)) # training session data
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
        batchLoss = fxpTensor.meanSquaredErrorLoss(Y_pred, target = batchY)
        batchLoss.name = 'batchLoss'
        batchAcc = fxpTensor.accuracy(Y_pred, batchY)
        fwdTime += time.time() - startTime
        startTime = time.time()
        batchLoss.backward()
        # update
        optimizer.step()
        bwdTime += time.time() - startTime
        # store loss and accuracy data
        trainLoss += fxpTensor.fxpToFloats(batchLoss.data) * batchInputIdxs.size
        trainAcc += batchAcc * batchInputIdxs.size
    # calculate average training loss and accuracy
    trainLoss /= numTrainInputs
    trainAcc /= numTrainInputs
    # validation
    startTime = time.time()
    Y_pred = forward(testX)
    testLoss = fxpTensor.meanSquaredErrorLoss(Y_pred, target = testY)
    testAcc = fxpTensor.accuracy(Y_pred, testY)
    fwdTime += time.time() - startTime
    outputFormat = 'Epoch {:4}:\n'\
                   'Train: Loss {:7.4f}\tAccuracy {:6.4f}\n'\
                   'Test : Loss {:7.4f}\tAccuracy {:6.4f}\n'\
                   'Time : Forward {:5.2f}s\tBackward {:5.2f}s'
    # note format of training session data
    trainingSessionData[epoch] = trainLoss, trainAcc, fxpTensor.fxpToFloats(testLoss.data), testAcc, fwdTime, bwdTime
    print(outputFormat.format(epoch, *trainingSessionData[epoch]))

# show accumulated overflow stats
fxpTensor.printOverflowStatistics()

# dump model
os.makedirs('./models/', exist_ok = True)
modelFile = './models/' + datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
paramDict = {param.name : param.data for param in params}
np.savez_compressed('{}.npz'.format(modelFile), **paramDict)
print('Model saved to {}.npz'.format(modelFile))
np.savetxt('{}.csv'.format(modelFile), trainingSessionData, delimiter = ',')
print('Training session data saved to {}.csv'.format(modelFile))
