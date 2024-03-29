#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:38:09 2022

@author: alex
"""
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
import pandas as pd
from modelEvaluation.predict import predict
from sklearn.metrics import mean_squared_error,average_precision_score
from modelEvaluation.compare import performance
from dataManipulation.dataPreparation import getData
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression,LogisticRegression
from pathlib import Path
import os
import re
import joblib
#%%
def transform(X):
    dff=X.copy()
    quantiles=dff['GMA_peso-ip'].quantile([0.5,0.8,0.95,0.99])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.50],2,1)
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.80],3,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.95],4,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.99],5,dff['complejidad'])
    dff['GMA']=dff.filter(regex='GMA_[0-9]+').idxmax(axis=1)
    dff['Nueva_categoria']=dff.GMA.str.slice(0,-1)+dff.complejidad.astype(str)
    dff=dff[[c for c in dff if not (('GMA' in c))]]
    dff=pd.concat([dff, pd.get_dummies(dff.Nueva_categoria)],axis=1)
    dff.drop(['complejidad','Nueva_categoria'], axis=1,inplace=True)
    return dff

def reproduce_GMA_complexity(X):
    dff=X.copy()
    dff['GMA']=dff.filter(regex='GMA_[0-9]+').idxmax(axis=1)
    dff['complejidad']=1
    
    cutoff={'GMA_10':[1.381,2.23,3.247,5.38],
           'GMA_20':[4.543,8.469,11.514,15.506],
           'GMA_31':[0.97,1.919,2.846,4.575],
           'GMA_32':[2.842,4.562,6.183,8.949],
           'GMA_33':[8.442,13.317,18.38,28.506],
           'GMA_40':[10.264,18.772,27.755,42.25]} 
    
    for gmagroup in sorted(dff.GMA.str.slice(0,-1).unique()):
        in_group=dff.GMA.str.slice(0,-1)==gmagroup
        quantiles_=cutoff[gmagroup]
        # quantiles=[0,1,2,3]
        dff.loc[in_group,'complejidad']=np.where(dff.loc[in_group,'GMA_peso-ip']>=quantiles_[0],2,1)
        dff.loc[in_group,'complejidad']=np.where(dff.loc[in_group,'GMA_peso-ip']>=quantiles_[1],3,dff.loc[in_group,'complejidad'])
        dff.loc[in_group,'complejidad']=np.where(dff.loc[in_group,'GMA_peso-ip']>=quantiles_[2],4,dff.loc[in_group,'complejidad'])
        dff.loc[in_group,'complejidad']=np.where(dff.loc[in_group,'GMA_peso-ip']>=quantiles_[3],5,dff.loc[in_group,'complejidad'])
    dff['Nueva_categoria']=dff.GMA.str.slice(0,-1)+dff.complejidad.astype(str)
    dff=dff[[c for c in dff if not (('GMA' in c))]]
    dff=pd.concat([dff, pd.get_dummies(dff.Nueva_categoria)],axis=1)
    dff.drop(['complejidad','Nueva_categoria'], axis=1,inplace=True)
    return dff
#%%
table=pd.DataFrame()
predictors=['PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG_','PATIENT_ID|FEMALE|AGE_[0-9]+$|GMA_']
CCS=[False, True]
GMACATEGORIES=[False,True]

for predictors_,CCS_,GMACATEGORIES_ in zip(predictors,CCS,GMACATEGORIES):
    preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
    modelname=os.path.join(config.MODELPATH,f'linear_{preds}.joblib' )
    if not Path(modelname).is_file():
        X,y=getData(2016,
                predictors=predictors_,
                exclude=None,
                resourceUsage=False,
                FULLEDCS=False,
                CCS=CCS_,
                PHARMACY=False,
                BINARIZE_CCS=False,
                GMACATEGORIES=GMACATEGORIES_
                )
        y=y[config.COLUMNS]

        if config.ALGORITHM=='logistic':
            y=np.where(y>0,1,0)
        if config.ALGORITHM=='linear':
            estimator=LinearRegression(n_jobs=-1)
        else:
            estimator=LogisticRegression(n_jobs=-1,penalty='none')
        X.drop('PATIENT_ID',axis=1,inplace=True)
        t0=time()
        fit=estimator.fit(X, y)
        print('fitting time: ',time()-t0)
        preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
        util.savemodel(config, fit, name=f'{config.ALGORITHM}_{preds}')

#%%
predictions={}
for predictors_,CCS_,GMACATEGORIES_ in zip(predictors,CCS,GMACATEGORIES):
    X,y=getData(2017,
            predictors=predictors_,
            exclude=None,
            resourceUsage=False,
            FULLEDCS=False,
            CCS=CCS_,
            PHARMACY=False,
            BINARIZE_CCS=False,
            GMACATEGORIES=GMACATEGORIES_,
            )
    
    preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
    model=joblib.load(os.path.join(config.MODELPATH,f'{config.ALGORITHM}_{preds}.joblib' ))
    predictions[preds]=predict(f'{config.ALGORITHM}_{preds}',config.EXPERIMENT,2018,X=X,y=y)


#%%
""" BUILD MODEL WITH OUR OWN COMPLEXITY CATEGORIES """
X,y=getData(2016,
        predictors='PATIENT_ID|FEMALE|AGE_[0-9]+$|GMA_',
        exclude=None,
        resourceUsage=False,
        FULLEDCS=False,
        CCS=True,
        PHARMACY=False,
        BINARIZE_CCS=False,
        GMACATEGORIES=True,
        additional_columns=['GMA_peso-ip']
        )
y=y[config.COLUMNS]

if config.ALGORITHM=='logistic':
    y=np.where(y>0,1,0)

X=transform(X)
X.drop('PATIENT_ID',axis=1,inplace=True)
if config.ALGORITHM=='linear':
    estimator=LinearRegression(n_jobs=-1)
else:
    estimator=LogisticRegression(n_jobs=-1,penalty='none')

t0=time()
fit=estimator.fit(X, y)
print('fitting time: ',time()-t0)
preds=re.sub('[^a-zA-Z]|PATIENT_ID','',predictors_)
util.savemodel(config, fit, name=f'{config.ALGORITHM}_{preds}_complejidadcuantiles')

#%%PREDICT FOR THIS MODEL
X,y=getData(2017,
        predictors='PATIENT_ID|FEMALE|AGE_[0-9]+$|GMA_',
        exclude=None,
        resourceUsage=False,
        FULLEDCS=False,
        CCS=True,
        PHARMACY=False,
        BINARIZE_CCS=False,
        GMACATEGORIES=True,
        additional_columns=['GMA_peso-ip']
        )

X=transform(X)

if config.ALGORITHM=='logistic':
    y[config.COLUMNS]=np.where(y[config.COLUMNS]>0,1,0)
    
model=joblib.load(os.path.join(config.MODELPATH,f'{config.ALGORITHM}_{preds}_complejidadcuantiles.joblib' ))
predictions[f'{preds}_complejidadcuantiles']=predict(f'{config.ALGORITHM}_{preds}_complejidadcuantiles',config.EXPERIMENT,2018,X=X,y=y)

#%%

for model, predictions_ in predictions.items():
    preds=predictions_[0]
    score=predictions_[1]
    recall, ppv, _, _ = performance(obs=preds.OBS, pred=preds.PRED, K=20000)
    if config.ALGORITHM=='logistic':
        ap=average_precision_score(np.where(preds.OBS>0,1,0), preds.PRED)
        table=pd.concat([table,
                        pd.DataFrame.from_dict({'Model':[model], 'AUC':[ score], 'AP':[ap],
                         'R@20k': [recall], 'PPV@20K':[ppv]})]) 
    else:
        rmse=mean_squared_error(preds.OBS,preds.PRED, squared=False)
        table=pd.concat([table,
                         pd.DataFrame.from_dict({'Model':[model], 'R2':[ score], 'RMSE':[rmse],
                          'R@20k': [recall], 'PPV@20K':[ppv]})])
    
#%%
print(table.to_markdown(index=False))
"""
| Model                             |       R2 |    RMSE |   R@20k |   PPV@20K |
|:----------------------------------|---------:|--------:|--------:|----------:|
| FEMALEAGEACG                      | 0.140944 | 3739.57 | 0.14565 |  0.1416   |
| FEMALEAGEGMA                      | 0.140171 | 3741.25 | 0.1642  |  0.162655 |
| FEMALEAGEGMA_complejidadcuantiles | 0.140481 | 3740.57 | 0.17495 |  0.156373 |

| Model                             |      AUC |       AP |     R@20k |   PPV@20K |
|:----------------------------------|---------:|---------:|----------:|----------:|
| FEMALEAGEACG                      | 0.756804 | 0.215594 | 0.0595967 |  0.41457  |
| FEMALEAGEGMA                      | 0.768863 | 0.219586 | 0.0747574 |  0.431335 |
| FEMALEAGEGMA_complejidadcuantiles | 0.770331 | 0.225687 | 0.0680096 |  0.477815 |
"""