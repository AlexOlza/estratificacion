#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:43:11 2023

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
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder

from main.gma_trials.forward_regression import sklearn_forward_regression,sklearn_stepwise_regression
import os
from modelEvaluation.predict import predict
from modelEvaluation.compare import performance
from sklearn.metrics import mean_squared_error
#%%
import joblib
X,y=getData(2017,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=True,
             GMACATEGORIES=True,
             GMA_DROP_DIGITS=0,
             additional_columns=[])
X_ordinal=X.copy()
X_ordinal=reverse_one_hot(X_ordinal)
#%%
descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
minimal=list(['FEMALE'])+list([c for c in X if (('AGE' in c) or ('GMA' in c))])
# to_add=['','|CCS','|PHARMA','','|CCS','|PHARMA',]
modelnames_=['AGE_FEMALE_GMA','AGE_FEMALE_GMA_CCS','AGE_FEMALE_GMA_CCS_PHARMA']
modelnames=list([f'stepwise_ordinal_{n}' for n in modelnames_])+list([f'local_stepwise_ordinal_{n}' for n in modelnames_]))
predictions={}
df={}
for name in modelnames:
    model=joblib.load(os.path.join(config.MODELPATH,f'{name}.joblib'))
    features=model.feature_names_in_
    df[name]=pd.DataFrame({'CATEGORIES':features})
    df[name]=pd.merge(df[name],descriptions,on='CATEGORIES',how='left')
    try:
        predictions[name]=predict(f'{name}',config.EXPERIMENT,2018,
                                  X=X[list(model.feature_names_in_)+['PATIENT_ID']],y=y)
    except KeyError:
        predictions[name]=predict(f'{name}',config.EXPERIMENT,2018,
                                  X=X_ordinal[list(model.feature_names_in_)+['PATIENT_ID']],y=y)
    print(df[name].to_markdown(index=False))
    print('---'*10)
#%%

table=pd.DataFrame()
for model, predictions_ in predictions.items():
    preds=predictions_[0]
    R2=predictions_[1]
    recall, ppv, _, _ = performance(obs=preds.OBS, pred=preds.PRED, K=20000)
    rmse=mean_squared_error(preds.OBS,preds.PRED, squared=False)
    table=pd.concat([table,
                     pd.DataFrame.from_dict({'Model':[model], 'R2':[ R2], 'RMSE':[rmse],
                      'R@20k': [recall], 'PPV@20K':[ppv]})])
    
#%%
print(table.to_markdown(index=False))