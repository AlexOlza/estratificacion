#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:23:21 2021

@author: aolza
"""
chosen_config=input('CONFIG FILENAME: ')#example logisticOLDBASE
chosen_config='configurations.local.'+chosen_config
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
import os
import numpy as np
import pandas as pd
from main import SafeLogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016,oldbase=False)#new data 
#%%

y=np.where(y.urg>=1,1,0)
print('Sample size ',len(X))

#%%
logistic=SafeLogisticRegression(penalty='none',max_iter=1000,verbose=0)

to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')
from time import time
t0=time()
fit=logistic.safeFit(X, y)
print('fitting time: ',time()-t0)
#%%
from configurations.security import savemodel
savemodel(config, fit)

