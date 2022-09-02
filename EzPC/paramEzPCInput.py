#! /usr/bin/env python3

from __future__ import annotations
import numpy as np

# generation params
Shift = 24
NumClasses = 10
NumFeatures = 16
HiddenDim = 10
RankW = 4
RankU = 0

# RNG seeding
rng = np.random.default_rng(0)

# input weights
if RankW:
    Wr = rng.normal(0, 0.1, (NumFeatures, RankW))
    inDimW = RankW
else:
    inDimW = NumFeatures
Wf = rng.normal(0, 0.1, (inDimW, HiddenDim))
Wi = rng.normal(0, 0.1, (inDimW, HiddenDim))
Wc = rng.normal(0, 0.1, (inDimW, HiddenDim))
Wo = rng.normal(0, 0.1, (inDimW, HiddenDim))
# hidden state weights
if RankU:
    Ur = rng.normal(0, 0.1, (HiddenDim, RankU))
    inDimU = RankU
else:
    inDimU = HiddenDim
Uf = rng.normal(0, 0.1, (inDimU, HiddenDim))
Ui = rng.normal(0, 0.1, (inDimU, HiddenDim))
Uc = rng.normal(0, 0.1, (inDimU, HiddenDim))
Uo = rng.normal(0, 0.1, (inDimU, HiddenDim))
# biases
Bf = rng.normal(0, 0.1, (HiddenDim,))
Bi = rng.normal(0, 0.1, (HiddenDim,))
Bc = rng.normal(0, 0.1, (HiddenDim,))
Bo = rng.normal(0, 0.1, (HiddenDim,))
# final fully-connected layer params
Wl = rng.normal(0, 0.1, (HiddenDim, NumClasses))
Bl = rng.normal(0, 0.1, (NumClasses,))

# list of params in order of storage
params: list[np.ndarray] = []
if RankW:
    params.append(Wr)
params += [Wf, Wi, Wc, Wo]
if RankU:
    params.append(Ur)
params += [Uf, Ui, Uc, Uo, Bf, Bi, Bc, Bo, Wl, Bl]

outfile = open('autogen/params.server', 'w')
for param in params:
    fxpParam = np.round(param * (2 ** Shift)).astype(int)
    np.savetxt(outfile, fxpParam, fmt = '%d')
outfile.close()
