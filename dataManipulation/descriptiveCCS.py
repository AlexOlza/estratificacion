#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:15:43 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
year=int(sys.argv[3])

import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
import numpy as np
from dataManipulation.dataPreparation import getData, generateCCSData
#%%

X,y=getData(year)
#%%
n,p={},{}
for c in X.drop('PATIENT_ID',axis=1):
    n[c]= X[c].sum()
    
#%%

for c in X.drop('PATIENT_ID',axis=1):
    p[c]= sum(np.where(X[c]>=1,1,0))/len(X)

#%%
from collections import Counter
K=15
c = Counter(n)
cp= Counter(p)
most_common = c.most_common(K) 
most_prevalent= cp.most_common(K) 

print('MOST COMMON CCSs:')
print(most_common)

print('MOST PREVALENT CCSs:')