#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:23:21 2021

@author: aolza
"""
chosen_config=input('CONFIG FILENAME: ')#example configurations.local.logistic
import importlib
importlib.invalidate_caches()
# chosen_config=importlib.import_module(chosen_config,package='estratificacion')
from python_settings import settings as config
# from python_settings import SetupSettings
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
from sklearn import linear_model
from sklearn.model_selection import train_test_split
np.random.seed(config.SEED)

X,y=getData(2016,oldbase=False)
Xx,Yy=getData(2016)
# sum(Yy!=y.urg)
#%%
# assert len(y.columns)==1, 'This model is built for a single response variable! Modify config.COLUMNS'

# na_indices=X[X.isna().any(axis=1)].index
#there arer no nas here!!!
# X.drop(na_indices,axis=0,inplace=True)
# y.drop(na_indices,axis=0,inplace=True)
# y[config.COLUMNS]=np.where(y[config.COLUMNS]>=1,1,0)
Yy=np.where(Yy.to_numpy()>=1,1,0)
print('Sample size ',len(Xx))
# X_train, X_test, y_train, y_test = train_test_split(Xx,Yy,test_size=0.3,random_state=config.SEED)


#%%
logistic=linear_model.LogisticRegression(penalty='none',max_iter=1000,verbose=0)
# fit=logistic.fit(X_train,y_train)
# fit=logistic.fit(Xx, Yy.to_numpy().ravel())
to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        Xx.drop(c,axis=1,inplace=True)
    except:
        pass
from time import time
t0=time()
fit=logistic.fit(Xx, Yy.ravel())
print('fitting time: ',time()-t0)
#%%
from configurations.security import savemodel
savemodel(config, fit, comments='this is with Xx and Yy')
#%%
# 
import joblib 
oldmodel=joblib.load('/home/aolza/Desktop/estratificacion/models/OLDBASE/oldlog.joblib')
# newmodel=joblib.load('')
# print(oldmodel.coef_[0]==fit.coef_[0])
withX=joblib.load('/home/aolza/Desktop/estratificacion/models/OLDBASE/logistic20211201_180659.joblib')
oldpred=oldmodel.predict_proba(X.head())
withXxpred=fit.predict_proba(X.head())
withXpred=withX.predict_proba(X.head())
