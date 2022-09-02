#! /usr/bin/env python3

# compare float and fixed point models to calculate disagreement ratio
# disagreement ratio: fraction of test samples on which the two models disagree (even if both are wrong)

from __future__ import annotations
import numpy as np
import sys

import fxpTensor
import myTensor

# input info
inputDir = 'data/usps10/'
numClasses = 10
numFeats = 16
timeSteps = 16
# training params
hiddenDim = 10
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
testX, _ = processData(inputDir + 'test.npy')

def getY(filename: str, lib) -> np.ndarray:
    model = np.load(filename)
    Wf = lib.Tensor(model['Wf'])
    Wi = lib.Tensor(model['Wi'])
    Wo = lib.Tensor(model['Wo'])
    Wc = lib.Tensor(model['Wc'])
    Uf = lib.Tensor(model['Uf'])
    Ui = lib.Tensor(model['Ui'])
    Uo = lib.Tensor(model['Uo'])
    Uc = lib.Tensor(model['Uc'])
    bf = lib.Tensor(model['bf'])
    bi = lib.Tensor(model['bi'])
    bo = lib.Tensor(model['bo'])
    bc = lib.Tensor(model['bc'])
    Wl = lib.Tensor(model['Wl'])
    bl = lib.Tensor(model['bl'])
    if rankW:
        Wr = lib.Tensor(model['Wr'])
    if rankU:
        Ur = lib.Tensor(model['Ur'])
    X = lib.Tensor(testX, 'X')
    thisBatchSize = X.data.shape[2]
    bExpander = lib.Tensor(np.ones((1, thisBatchSize), dtype = float), 'bX')
    H = lib.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'H0')
    C = lib.Tensor(np.zeros((hiddenDim, thisBatchSize), dtype = float), 'C0')
    for timeStep in range(timeSteps):
        Xt = X[timeStep]
        if rankW:
            Xt = lib.matmul(Wr, Xt)
        if rankU:
            H = lib.matmul(Ur, H)
        Gf = lib.sigmoid(lib.sfmadd(Wf, Xt, Uf, H, bf, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Gi = lib.sigmoid(lib.sfmadd(Wi, Xt, Ui, H, bi, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Go = lib.sigmoid(lib.sfmadd(Wo, Xt, Uo, H, bo, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        Gc = lib.tanh(lib.sfmadd(Wc, Xt, Uc, H, bc, bExpander, ops = [np.matmul, np.matmul, np.matmul]))
        newC = lib.sfmadd(C, Gf, Gc, Gi, ops = [np.multiply, np.multiply])
        newH = Go * lib.tanh(newC)
        # feedback
        H = newH
        C = newC
    # final layer
    Y_pred = lib.sfmadd(Wl, H, bl, bExpander, ops = [np.matmul, np.matmul])
    return Y_pred.data

Y_float = getY(sys.argv[1], myTensor)
Y_float = np.argmax(Y_float, axis = 0)
Y_fxp = getY(sys.argv[2], fxpTensor)
Y_fxp = np.argmax(Y_fxp, axis = 0)

disagreements = Y_float != Y_fxp
disagreeRatio = np.count_nonzero(disagreements) / Y_float.size
print(disagreeRatio)
