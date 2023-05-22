#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:52:08 2023

@author: alex
"""

prefix='stepwise_'
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

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

from main.gma_trials.forward_regression import sklearn_stepwise_regression_simple
#%%
X,y=getData(2016,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=False,
             GMACATEGORIES=False,
             GMA_DROP_DIGITS=0,
             additional_columns=[])

print('Sample size ',len(X))

#%%
from configurations import utility as util
import re
ycol=config.COLUMNS[0]
df=pd.merge(X,y,on='PATIENT_ID').drop('PATIENT_ID',axis=1)#.sample(100)
#%%
minimal=list(['FEMALE'])+list([c for c in X if ('AGE' in c)])
to_add=['','|CCS','|PHARMA']
modelnames=['AGE_FEMALE','AGE_FEMALE_CCS','AGE_FEMALE_CCS_PHARMA']
for group, name in zip(to_add,modelnames):
    cand='|'.join(minimal)+group
    print('MINIMAL VARIABLES: ')
    print(minimal)
    print('POTENTIAL CANDIDATES: ',cand)
    model=sklearn_stepwise_regression_simple(df.filter(regex=cand+f'|{ycol}'),
                                      minimal=minimal,
                                      y=ycol)
    minimal=model.feature_names_in_
    print('----'*10)
    print('\n')
    util.savemodel(config, model, name=f'{prefix}{name}')

#%%
import joblib
X,y=getData(2017,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=False,
             GMACATEGORIES=False,
             GMA_DROP_DIGITS=0,
             additional_columns=[])

#%%
import os
from modelEvaluation.predict import predict
descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
minimal=list(['FEMALE'])+list([c for c in X if ('AGE' in c)])
to_add=['','|CCS','|PHARMA']
predictions={}
for group, name in zip(to_add,modelnames):
    model=joblib.load(os.path.join(config.MODELPATH,f'{prefix}{name}.joblib'))
    features=model.feature_names_in_
    df=pd.DataFrame({'CATEGORIES':features})
    df=pd.merge(df,descriptions,on='CATEGORIES',how='left')
    
    predictions[name]=predict(f'{prefix}{name}',config.EXPERIMENT,2018,
                              X=X[list(model.feature_names_in_)+['PATIENT_ID']],y=y)
print(df.to_markdown(index=False))
#%%
from modelEvaluation.compare import performance
from sklearn.metrics import mean_squared_error
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