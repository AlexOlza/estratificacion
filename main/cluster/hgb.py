#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:55:09 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
chosen_config='configurations.cluster.'+sys.argv[1]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import RandomizedSearchCV

np.random.seed(config.SEED)

pred16,y17=getData(2016,predictors=True)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
print('Sample size ',len(pred16))

#%%
"""BALANCEO LAS CLASES Y TRAIN-TEST SPLIT"""
idx=np.array(list(itertools.chain.from_iterable((y17[config.COLUMNS] >= 1).values)))
ing_indices = y17.loc[idx].index
noing_indices = y17.loc[(~idx)].index
half_sample_size = len(ing_indices)  # Equivalent to len(data[data.Healthy == 0])
random_indices = np.random.choice(noing_indices, half_sample_size, 
                                  replace=False)

X=pd.concat([pred16.loc[random_indices],pred16.loc[ing_indices]])
na_indices=X[X.isna().any(axis=1)].index
X.drop(na_indices,axis=0,inplace=True)
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

y=pd.concat([y17.loc[random_indices],y17.loc[ing_indices]])
y.drop(na_indices,axis=0,inplace=True)
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Dropping NA. Amount: ',len(na_indices))
print('Sample size ',len(X))
#%%
forest = RandomizedSearchCV(estimator = config.FOREST, 
                               param_distributions = config.RANDOM_GRID,
                               n_iter = 3,
                               cv = 2, 
                               verbose=2,
                               random_state=config.SEED,
                               n_jobs =-1)
forest.fit(X,y)

#%%
""" SAVE TRAINED MODEL """
util.savemodel(config,forest.best_estimator_)
""" PERFORMANCE """
from sklearn.metrics import roc_auc_score
probs=forest.predict_proba(X)[:,1]
print('auc in training data ',roc_auc_score(y,probs))
