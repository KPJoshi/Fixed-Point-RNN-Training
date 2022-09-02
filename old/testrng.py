#! /usr/bin/env python3

def perm(ep, ns):
 indices = []
 for i in range(ns):
  indices.append(i)
 out = []
 for i in range(ns):
  sel = ((ep+1)*(i+1)*257)%(ns-i)
  out.append(indices[sel])
  indices[sel] = indices[ns-i-1]
 return out

for i in range(20):
 print(perm(i,10))
