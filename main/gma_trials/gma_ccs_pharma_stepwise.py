#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

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

from dataManipulation.dataPreparation import getData, reverse_one_hot
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from main.gma_trials.forward_regression import sklearn_stepwise_regression

X,y=getData(2016,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=True,
             GMACATEGORIES=True,
             GMA_DROP_DIGITS=0,
             additional_columns=[])

# X=reverse_one_hot(X)
# assert False
print('Sample size ',len(X))

#%%
from configurations import utility as util
import re
ycol=config.COLUMNS[0]
df=pd.merge(X,y,on='PATIENT_ID').drop('PATIENT_ID',axis=1)#.sample(100)
minimal=list(['FEMALE'])+list([c for c in X if (('AGE' in c) or ('GMA' in c))])
to_add=['','|CCS','|PHARMA']
modelnames=['AGE_FEMALE_GMA','AGE_FEMALE_GMA_CCS','AGE_FEMALE_GMA_CCS_PHARMA']
for group, name in zip(to_add,modelnames):
    cand='|'.join(minimal)+group
    print('MINIMAL VARIABLES: ')
    print(minimal)
    print('POTENTIAL CANDIDATES: ',cand)
    model=sklearn_stepwise_regression(df.filter(regex=cand+f'|{ycol}'),
                                      minimal=minimal,
                                      y=ycol,
                                      tol=1e-3)
    minimal=model.feature_names_in_
    print('----'*10)
    print('\n')
    util.savemodel(config, model, name=f'stepwise_{name}')

