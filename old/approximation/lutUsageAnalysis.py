#! /usr/bin/env python3

# analyses LUT usage data recorded by the fixpoint library
# displays LUT usage graphs

import matplotlib.pyplot as plt
import numpy as np
import os

dtype = np.int16
shift = 12
statsFile = '../data/LUTUsage.npz'
outDir = 'lutUsagePlots'

lutDataDict = np.load(statsFile)
os.makedirs(outDir, exist_ok = True)
bits = np.iinfo(dtype).bits
X = np.arange(2 ** bits, dtype = np.int16).astype(float) / (2 ** shift)

for lutName, lutData in lutDataDict.items():
    count = np.sum(lutData)
    mean = np.average(X, weights = lutData)
    std = np.sqrt(np.average((X - mean) ** 2, weights = lutData))
    present = X[lutData > 0]
    minimum = np.min(present)
    maximum = np.max(present)
    mode = X[np.argmax(lutData)]
    print('Generating plot for', lutName)
    fig, ax = plt.subplots(constrained_layout = True)
    fig.suptitle(lutName)
    ax.scatter(X, lutData, marker = '.')
    statsString = 'Accesses: {:10}\nAccessed: {:10}\nMode: {:7.4f}\nMean: {:7.4f}\n Std: {:7.4f}\n Min: {:7.4f}\n Max: {:7.4f}'.format(count, present.size, mode, mean, std, minimum, maximum)
    ax.text(-8, np.max(lutData), statsString, {'fontfamily': 'monospace'}, horizontalalignment = 'left', verticalalignment = 'top')
    ax.set_xlabel('Input')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    plt.savefig(outDir + '/' + lutName + '.png')
    plt.close()
