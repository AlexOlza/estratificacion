#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

@author: aolza
"""

#%%
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

from dataManipulation.dataPreparation import getData, reverse_one_hot
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder

from main.gma_trials.forward_regression import sklearn_forward_regression,sklearn_stepwise_regression

X,y=getData(2016,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=True,
             GMACATEGORIES=True,
             GMA_DROP_DIGITS=0,
             additional_columns=[])

# X, y=X.sample(1000,random_state=1), y.sample(1000, random_state=1)
# y=y[config.COLUMNS]
X=reverse_one_hot(X)

print('Sample size ',len(X))

#%%
df=pd.merge(X,y,on='PATIENT_ID').drop('PATIENT_ID',axis=1)#.sample(100)
model=sklearn_stepwise_regression(df, config.COLUMNS[0])
#%%
from configurations import utility as util
util.savemodel(config, model, name='stepwise')
