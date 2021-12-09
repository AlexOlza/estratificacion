#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:19:44 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.cluster import configRandomForest as randomForest_settings
if not config.configured:
    config.configure(randomForest_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
import os
import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
np.random.seed(config.SEED)

pred16,y17=getData(2016,predictors=True)
assert len(y17.columns)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
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

y=pd.concat([y17.loc[random_indices],y17.loc[ing_indices]])
y.drop(na_indices,axis=0,inplace=True)
y[config.COLUMNS]=np.where(y[config.COLUMNS]>=1,1,0)
print('Dropping NA')
print('Sample size ',len(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=config.SEED)



forest = RandomizedSearchCV(estimator = config.FOREST, 
                               param_distributions = config.RANDOM_GRID,
                               n_iter = config.N_ITER,
                               cv = config.CV, 
                               verbose=0,
                               random_state=config.SEED,
                               n_jobs =-1)
forest.fit(X_train,y_train)
#%%
""" SAVE TRAINED MODEL """
from configurations.security import savemodel
savemodel(config,forest)
#
