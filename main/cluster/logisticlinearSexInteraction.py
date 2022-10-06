#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:19:12 2022

@author: aolza
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

from dataManipulation.dataPreparation import getData
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016) 

X.drop('PATIENT_ID', axis=1, inplace=True)

features=X.columns
interactions= X.drop([ 'FEMALE'], axis=1).multiply(X.FEMALE,axis=0).astype(np.int8)
interactions.rename(columns={c:f'{c}INTsex' for c in interactions}, inplace=True)
X=pd.concat([X,interactions],axis=1)

print('Number of columns: ', len(X.columns))
#%%

print('Sample size ',len(X))
# assert False
#%%

if config.ALGORITHM=='logistic':
    y=np.where(y[config.COLUMNS]>=1,1,0)
    y=y.ravel()
    estimator=LogisticRegression(penalty='none',max_iter=1000,verbose=0, warm_start=False)
elif config.ALGORITHM=='linear':
    y=y[config.COLUMNS]
    estimator=LinearRegression(n_jobs=-1)
else:
    assert False, 'This script is only suitable for linear and logistic algorithms. Check your configuration!'


from time import time
t0=time()
fit=estimator.fit(X, y)
print('fitting time: ',time()-t0)
#%%

config.ALGORITHM=f'{config.ALGORITHM}SexInteraction'
import os

util.savemodel(config, fit, name=config.ALGORITHM)

