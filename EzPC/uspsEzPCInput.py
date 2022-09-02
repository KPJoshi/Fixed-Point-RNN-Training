#! /usr/bin/env python3

import numpy as np

inputdir = '../data/usps10/'
outputdir = 'autogen/'
shift = 24
numClasses = 10

trainraw = np.load(inputdir + 'train.npy')
testraw = np.load(inputdir + 'test.npy')

def process(arr):
  numSamples = arr.shape[0]
  labels = arr[:,0] - 1
  labels = labels.astype(int)
  denseLabels = np.zeros((numSamples, numClasses), dtype = int)
  denseLabels[np.arange(numSamples), labels] = 1
  data = arr[:,1:]
  data = np.round(data * (2 ** shift)).astype(int)
  return np.column_stack((denseLabels, data))

outfile = open(outputdir + 'traintest.client', 'w')
trainData = process(trainraw)
np.savetxt(outfile, trainData, fmt = '%d')
outfile.write('\n')
testData = process(testraw)
np.savetxt(outfile, testData, fmt = '%d')
outfile.close()
