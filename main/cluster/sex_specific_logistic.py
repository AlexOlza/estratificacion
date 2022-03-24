#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN TWO SEPARATE LOGISTIC REGRESSION MODELS FOR MALES AND FEMALES
Created on Fri Mar 18 12:50:05 2022

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
#%%
female=X['FEMALE']==1
male=X['FEMALE']==0
sex=[ 'Mujeres','Hombres']

for group, groupname in zip([female,male],sex):
    print(groupname)
    Xgroup=X.loc[group]
    ygroup=y.loc[group]
    print(Xgroup.PATIENT_ID)
    print(ygroup)
    assert (all(Xgroup['FEMALE']==1) or all(Xgroup['FEMALE']==0))
    ygroup=np.where(ygroup[config.COLUMNS]>=1,1,0)
    ygroup=ygroup.ravel()
    print('Sample size ',len(Xgroup), 'positive: ',sum(ygroup))
    logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0, warm_start=False)
    
    to_drop=['PATIENT_ID','ingresoUrg', 'FEMALE']
    for c in to_drop:
        try:
            Xgroup.drop(c,axis=1,inplace=True)
            util.vprint('dropping col ',c)
        except:
            pass
            util.vprint('pass')
    from time import time
    t0=time()
    fit=logistic.fit(Xgroup, ygroup)
    print('fitting time: ',time()-t0)
    util.savemodel(config, fit,  name=f'logistic{groupname}')

