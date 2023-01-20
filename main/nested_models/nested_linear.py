#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:23:21 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
try: 
    chosen_config='configurations.cluster.'+sys.argv[1]
    experiment='configurations.'+sys.argv[2]
except ValueError:
    chosen_config='configurations.cluster.logistic'
    experiment='configurations.cost'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 
config.ALGORITHM='linear'
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os
from time import time
import pandas as pd
#%%
np.random.seed(config.SEED)

X,y=getData(2016,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=True,
             GMACATEGORIES=True,
             GMA_DROP_DIGITS=0,
             additional_columns=[]) 
y=y.COSTE_TOTAL_ANO2
#%%
linear=LinearRegression(n_jobs=-1)

to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')


if (not 'ACG' in config.PREDICTORREGEX):
    if (hasattr(config, 'PHARMACY')):
        CCSPHARMA='PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS|PHARMA' if config.PHARMACY else None
    else: CCSPHARMA= None
    if (hasattr(config, 'GMA')):
        CCSGMA='PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS|PHARMA|GMA' if config.GMA else None
    else: CCSGMA= None
else: 
    CCSPHARMA=None
    CCSGMA=None

variables={'Demo':'PATIENT_ID|FEMALE|AGE_[0-9]+$',
           'DemoDiag':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_' if 'ACG' in config.PREDICTORREGEX else 'PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS',
           'DemoDiagPharma':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_|RXMG_' if 'ACG' in config.PREDICTORREGEX else CCSPHARMA,
           'DemoDiagPharmaIsomorb':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_(?!NUR11|RES10)|RXMG_(?!ZZZX000)|ACG_' if 'ACG' in config.PREDICTORREGEX else CCSGMA
           }
#%%

for key, val in variables.items():
    print('STARTING ',key, val)
    if not val:
        continue
    if Path(os.path.join(config.MODELPATH,f'{key}.joblib')).is_file(): #the model is already there
        continue
    Xx=X.filter(regex=val, axis=1)

    print('Number of predictors: ',len(Xx.columns))
    logistic=LinearRegression(n_jobs=-1) 
    t0=time()
    fit=logistic.fit(Xx, y)
    print('fitting time: ',time()-t0)

    util.savemodel(config, fit, name='{0}'.format(key))


#%%
X, y=getData(2017)
X=X[[c for c in X if X[c].max()>0]]
PATIENT_ID=X.PATIENT_ID
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
   
y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y

#%%
import joblib
from modelEvaluation.predict import predict
from modelEvaluation.compare import performance
from sklearn.metrics import mean_squared_error
print('JOBLIB VERSION',joblib.__version__)
table=pd.DataFrame()
for key, val in variables.items():
    Xx=X.copy()
    if not val:
        continue
    if key=='DemoDiagPharmaBinary':
        print(Xx.PHARMA_Transplant.describe())
        Xx[[c for c in Xx if c.startswith('PHARMA')]]=(Xx[[c for c in Xx if c.startswith('PHARMA')]]>0).astype(int)
        print(Xx.PHARMA_Transplant.describe())
    try:
        preds,score=predict(key,experiment_name=config.EXPERIMENT,year=2018,
                          X=Xx.filter(regex=val, axis=1), y=y)
        recall, ppv, spec, newpred = performance(obs=preds.OBS, pred=preds.PRED, K=20000)

        rmse=mean_squared_error(preds.OBS,preds.PRED, squared=False)
        table=pd.concat([table,
                         pd.DataFrame.from_dict({'Model':[key], 'R2':[ score], 'RMSE':[rmse],
                          'R@20k': [recall], 'PPV@20K':[ppv]})])
    except:
        print(key , 'Failed')

#%%
print(table.to_markdown(index=False))