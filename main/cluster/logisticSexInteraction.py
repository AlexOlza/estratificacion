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
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016) 

X.drop('PATIENT_ID', axis=1, inplace=True)

features=X.columns

for column in features:
    if column!='FEMALE':
        X[f'{column}INTsex']=X[column]*X['FEMALE']

print('Number of columns: ', len(X.columns))
#%%
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Sample size ',len(X), 'positive: ',sum(y))
# assert False
#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

from time import time
t0=time()
fit=logistic.fit(X, y)
print('fitting time: ',time()-t0)
#%%

config.ALGORITHM='logisticSexInteraction'
import os

util.savemodel(config, fit, name='logisticSexInteraction')

