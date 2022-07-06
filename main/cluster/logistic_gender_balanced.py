#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:59:43 2022

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
def gender_balanced_undersampling(X,y):
    import itertools
    import pandas as pd
    XFemale=X.loc[X.FEMALE==1]
    yFemale=y.loc[XFemale.index]
    XMale=X.loc[X.FEMALE==0]
    yMale=y.loc[XMale.index]
    
    idx=np.array(list(itertools.chain.from_iterable((yFemale[config.COLUMNS] >= 1).values)))
    ing_indices = yFemale.loc[idx].index
    noing_indices = yFemale.loc[(~idx)].index
    half_sample_size = len(ing_indices)  # Equivalent to len(data[data.Healthy == 0])
    random_indices = np.random.choice(noing_indices, half_sample_size, 
                                  replace=False)

    XFemaleBalanced=pd.concat([XFemale.loc[random_indices],XFemale.loc[ing_indices]])
    yFemaleBalanced=pd.concat([yFemale.loc[random_indices],yFemale.loc[ing_indices]])
    
    idx=np.array(list(itertools.chain.from_iterable((yMale[config.COLUMNS] >= 1).values)))
    ing_indices = yMale.loc[idx].index
    noing_indices = yMale.loc[(~idx)].index
    half_sample_size = len(ing_indices)  # Equivalent to len(data[data.Healthy == 0])
    random_indices = np.random.choice(noing_indices, half_sample_size, 
                                  replace=False)
    
    XMaleBalanced=pd.concat([XMale.loc[random_indices],XMale.loc[ing_indices]])
    yMaleBalanced=pd.concat([yMale.loc[random_indices],yMale.loc[ing_indices]])
    return(XFemaleBalanced,yFemaleBalanced,XMaleBalanced,yMaleBalanced)

Xf,yf,Xm,ym=gender_balanced_undersampling(X,y)
#%%
assert False
print('first IDs (X)')
print(X.head().PATIENT_ID)
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

y=pd.concat([y17.loc[random_indices],y17.loc[ing_indices]])
y.drop(na_indices,axis=0,inplace=True)
print('first IDs (y)')
print(y.head().PATIENT_ID)
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Dropping NA. Amount: ',len(na_indices))
print('Sample size ',len(X))
np.random.seed(seed_hparam)
X,y=gender_balanced_undersampling(X,y)
#%%

y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Sample size ',len(X), 'positive: ',sum(y))

#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

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
fit=logistic.fit(X, y)
print('fitting time: ',time()-t0)
#%%
util.savemodel(config, fit,name='logistic_gender_balanced')

