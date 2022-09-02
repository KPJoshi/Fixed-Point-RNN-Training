#! /usr/bin/env bash

# calculate alphaT
sed -i 's/extern int64_t CalcAlphaT(int32_t iter);/int64_t CalcAlphaT(int32_t iter){return(int64_t)round(sqrt(1-pow((double)AdamBeta2\/FxpOne,iter))\/(1-pow((double)AdamBeta1\/FxpOne,iter))*0.01*FxpOne);}/g' build/lstm0.cpp

# delete 1D array
sed -i 's/extern void Free1D(auto& data);/void Free1D(auto\&A){A.clear();}/g' build/lstm0.cpp

# delete 2D array
sed -i 's/extern void Free2D(auto& data);/void Free2D(auto\&A){A.clear();}/g' build/lstm0.cpp
