#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:23:21 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
try: 
    chosen_config='configurations.cluster.'+sys.argv[1]
    experiment='configurations.'+sys.argv[2]
except ValueError:
    chosen_config='configurations.cluster.logistic'
    experiment='configurations.cost'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 
config.ALGORITHM='linear'
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import numpy as np
from sklearn.linear_model import LinearRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016,columns='COSTE_TOTAL_ANO2')#new data 
y=y.COSTE_TOTAL_ANO2
#%%
linear=LinearRegression()

to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')

variable_groups=[r'FEMALE|AGE_[0-9]+$','EDC_','RXMG_','ACG']
from time import time
for i in range(1,len(variable_groups)+1):
    regex=r'|'.join(variable_groups[:i])
    Xx=X.filter(regex=regex)
    print('Number of predictors: ',len(Xx.columns))
    linear=LinearRegression() 
    t0=time()
    fit=linear.fit(Xx, y)
    print('fitting time: ',time()-t0)
    util.savemodel(config, fit, name='nested_lin{0}'.format(i))
