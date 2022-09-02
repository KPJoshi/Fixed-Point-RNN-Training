#! /usr/bin/env python3

# compares approximation to actual function - replaced by iterativeapprox

import matplotlib.pyplot as plt
import numpy as np

from sigmoid import goldenFunc, fixPntFunc, functionName

dtype = np.int16
shift = 12

bits = np.iinfo(dtype).bits
dtypeMax = np.iinfo(dtype).max
shiftMult = 2 ** shift
fxpMax = dtypeMax / shiftMult
tests = (np.arange(2 * dtypeMax + 1, dtype = float) - dtypeMax) / shiftMult

print('Running tests...')
goldenVals = goldenFunc(tests)
fixPntVals = fixPntFunc(tests)

print('Calculating errors...')
# actual errors
errors = goldenVals - fixPntVals
absErrors = np.abs(errors)
maxAbsError = np.max(absErrors)
argmaxAbsError = tests[np.argmax(absErrors)]
avgError = np.mean(errors)
avgAbsError = np.mean(absErrors)
# ULP errors
fxpGoldenVals = np.round(goldenVals * shiftMult).astype(int)
ULPErrors = fxpGoldenVals - (fixPntVals * shiftMult).astype(int)
absULPErrors = np.abs(ULPErrors)
maxAbsULPError = np.max(absULPErrors)
argmaxAbsULPError = tests[np.argmax(absULPErrors)]
avgULPError = np.mean(ULPErrors)
avgAbsULPError = np.mean(absULPErrors)
print('Max abs actual error: {} at {}'.format(maxAbsError, argmaxAbsError))
print('Mean actual error: {}\nMean abs actual error: {}'.format(avgError, avgAbsError))
print('Max abs ULP error: {} at {}'.format(maxAbsULPError, argmaxAbsULPError))
print('Mean ULP error: {}\nMean abs ULP error: {}'.format(avgULPError, avgAbsULPError))

print('Plotting errors...')
xRange = np.ceil(fxpMax).astype(int)
xTicks = np.arange(2 * xRange + 1) - xRange
yRange = np.maximum(np.ceil(maxAbsError * shiftMult), maxAbsULPError).astype(int)
yTicks = np.arange(2 * yRange + 1) - yRange
fig, axs = plt.subplots(1, 2, figsize = (9, 4.5), constrained_layout = True)
fig.suptitle(functionName + ' Actual and ULP Error')
ax = axs[0]
ax.set_xlim(-xRange, xRange)
ax.set_xticks(xTicks)
ax.set_ylim(-yRange / shiftMult, yRange / shiftMult)
ax.set_yticks(yTicks / shiftMult)
ax.set_yticklabels(yTicks)
ax.set_xlabel('Input')
ax.set_ylabel('Actual Error')
ax.grid(True, which = 'both', linestyle = ':')
ax.scatter(tests, errors, color = 'red', s = 1)
formatString = 'Abs Max {:8.4f} @ {:8.4f}\nMean {:8.4f} / Abs Mean {:8.4f}'.format(maxAbsError, argmaxAbsError, avgError, avgAbsError)
ax.text(-xRange * .98, yRange / shiftMult * .98, formatString, {'fontfamily': 'monospace'}, horizontalalignment = 'left', verticalalignment = 'top')
ax = axs[1]
ax.set_xlim(-xRange, xRange)
ax.set_xticks(xTicks)
ax.set_ylim(-yRange, yRange)
ax.set_yticks(yTicks)
ax.set_xlabel('Input')
ax.set_ylabel('ULP Error')
ax.grid(True, which = 'both', linestyle = ':')
ax.scatter(tests, ULPErrors, color = 'red', s = 1)
formatString = 'Abs Max {:8d} @ {:8.4f}\nMean {:8.4f} / Abs Mean {:8.4f}'.format(maxAbsULPError, argmaxAbsULPError, avgULPError, avgAbsULPError)
ax.text(-xRange * .98, yRange * .98, formatString, {'fontfamily': 'monospace'}, horizontalalignment = 'left', verticalalignment = 'top')
plt.savefig('error.png')
