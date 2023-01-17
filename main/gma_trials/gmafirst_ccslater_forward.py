#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

@author: aolza
"""
import statsmodels.formula.api as sm
import pandas as pd

def forward_regression(df, y, candidates = ['AGE','GMA','FEMALE']):    
    ar2 = dict()
    last_max = -1
    
    while(True):
        for x in df.drop([y] + candidates, axis=1).columns:
            if len(candidates) == 0:
                features = x
            else:
                features = x + ' + '
                features += ' + '.join(candidates)
    
            model = sm.ols(y + ' ~ ' + features, df).fit()
            ar2[x] = model.rsquared
    
        max_ar2 =  max(ar2.values())
        max_ar2_key = max(ar2, key=ar2.get)
    
        if max_ar2 > last_max:
            candidates.append(max_ar2_key)
            last_max = max_ar2
    
            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Adjusted R2: ' + str(max_ar2))
            print('===============')
        else:
            print(model.summary())
            break
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
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
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
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
X=reverse_one_hot(X, integers=False)


enc = OrdinalEncoder(categories=[sorted(X.AGE.unique()),sorted(X.GMA.unique())],
                     dtype=np.int8)

X[['AGE','GMA']]=enc.fit_transform(X[['AGE','GMA']])

print('Sample size ',len(X))
assert False
#%%
df=pd.merge(X,y,on='PATIENT_ID').drop('PATIENT_ID',axis=1)#.sample(100)
forward_regression(df, config.COLUMNS[0])
