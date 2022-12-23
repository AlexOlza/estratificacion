#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:38:09 2022

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
import pandas as pd
from modelEvaluation.predict import predict
from sklearn.metrics import mean_squared_error
from modelEvaluation.compare import performance
from dataManipulation.dataPreparation import getData
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os
import re
import joblib
#%%
predictors=['PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG_','PATIENT_ID|FEMALE|AGE_[0-9]+$|GMA_']
CCS=[False, True]
GMACATEGORIES=[False,True]

for predictors_,CCS_,GMACATEGORIES_ in zip(predictors,CCS,GMACATEGORIES):
    X,y=getData(2016,columns='COSTE_TOTAL_ANO2',
            predictors=predictors_,
            exclude=None,
            resourceUsage=False,
            FULLEDCS=False,
            CCS=CCS_,
            PHARMACY=False,
            BINARIZE_CCS=False,
            GMACATEGORIES=GMACATEGORIES_
            )
    y=y.COSTE_TOTAL_ANO2

    linear=LinearRegression(n_jobs=-1)
    X.drop('PATIENT_ID',axis=1,inplace=True)
    t0=time()
    fit=linear.fit(X, y)
    print('fitting time: ',time()-t0)
    preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
    util.savemodel(config, fit, name=f'linear_{preds}')
#%%
predictions={}
for predictors_,CCS_,GMACATEGORIES_ in zip(predictors,CCS,GMACATEGORIES):
    X,y=getData(2017,columns='COSTE_TOTAL_ANO2',
            predictors=predictors_,
            exclude=None,
            resourceUsage=False,
            FULLEDCS=False,
            CCS=CCS_,
            PHARMACY=False,
            BINARIZE_CCS=False,
            GMACATEGORIES=GMACATEGORIES_
            )
    
    preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
    model=joblib.load(os.path.join(config.MODELPATH,f'linear_{preds}.joblib' ))
    predictions[preds]=predict(f'linear_{preds}',config.EXPERIMENT,2018,X=X,y=y)

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
"""
| Model        |       R2 |    RMSE |   R@20k |   PPV@20K |
|:-------------|---------:|--------:|--------:|----------:|
| FEMALEAGEACG | 0.140944 | 3739.57 | 0.14565 |  0.1416   |
| FEMALEAGEGMA | 0.140171 | 3741.25 | 0.1642  |  0.162655 |
"""