#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:32:34 2022

@author: aolza
Source: https://github.com/konosp/propensity-score-matching/blob/main/propensity_score_matching_v2.ipynb
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import pandas as pd
from dataManipulation.dataPreparation import getData
import numpy as np
import os
from time import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%%
Xx=X
z=Xx['FEMALE']
Xx=Xx.drop(['FEMALE', 'PATIENT_ID'], axis=1)
print('Sample size ',len(Xx), 'females: ',sum(z))
assert not 'AGE_85GT' in Xx.columns

  
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

t0=time()
fit=logistic.fit(Xx, z)
print('fitting time: ',time()-t0)

propensity_score=fit.predict_proba(Xx)[:,1]


IPTW=np.where(X.FEMALE==1,1/(propensity_score+1e-10), 1/(1-propensity_score+1e-10))
del Xx, propensity_score, z
#%%
y=np.where(y[config.COLUMNS]>=1,1,0)
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0,)
t0=time()
fit=logistic.fit(X.drop('PATIENT_ID', axis=1), y,sample_weight=IPTW)
print('fitting time: ',time()-t0)
#%%
util.savemodel(config, fit,name='logistic_IPTW')
#%%