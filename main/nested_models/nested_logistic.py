#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess the predictive performance gain
of the ACG variable, task #48 
Created on Mon Feb  7 12:30:42 2022

@author: aolza
What should I do with these? Ask Edu
HOSDOM|FRAILTY|INGRED_14GT
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
from time import time
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)#new data 
#%%
to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')

y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Sample size ',len(X), 'positive: ',sum(y))
# assert False
variable_groups=[r'FEMALE|AGE_[0-9]+$','EDC_','RXMG_','ACG']

for i in range(1,len(variable_groups)+1):
    regex=r'|'.join(variable_groups[:i])
    Xx=X.filter(regex=regex)
    print('Number of predictors: ',len(Xx.columns))
    logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0) 
    t0=time()
    fit=logistic.fit(Xx, y)
    print('fitting time: ',time()-t0)
    #%%
    util.savemodel(config, fit, name='nested_log{0}'.format(i))


