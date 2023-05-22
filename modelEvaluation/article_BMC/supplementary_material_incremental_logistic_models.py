#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:30:57 2023

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcms_excl_nbinj'
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
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from modelEvaluation.compare import performance
from dataManipulation.dataPreparation import getData
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import os
config.MODELPATH=os.path.join(config.MODELPATH,'nested_logistic/') if not 'nested_logistic' in config.MODELPATH else config.MODELPATH
config.PREDPATH=os.path.join(config.PREDPATH,'nested_logistic/') if not 'nested_logistic' in config.PREDPATH else config.PREDPATH
for direct in [config.MODELPATH, config.PREDPATH]:
    if not os.path.exists(direct):
        os.makedirs(direct)
        print('new dir ', direct)

#%%
np.random.seed(config.SEED)
prefix=''
X,y=getData(2016)

#%%
to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')

y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()


print('Sample size ',len(X), 'positive: ',sum(y))

variables={'AGESEX':'PATIENT_ID|FEMALE|AGE_[0-9]+$',
           'AGESEXACG':'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG_' ,
           'AGESEXACGEDC':'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG_|EDC_' ,
            # the full model is already computed
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
    logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0) 
    t0=time()
    fit=logistic.fit(Xx, y)
    print('fitting time: ',time()-t0)

    util.savemodel(config, fit, name='{0}'.format(key))


#%%
X, y=getData(2017)

PATIENT_ID=X.PATIENT_ID
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
   
y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y
#%%
import joblib
table=pd.DataFrame()
for key, val in variables.items():
    Xx=X.copy()
    if not val:
        continue

    model=joblib.load(os.path.join(config.MODELPATH,f'{key}.joblib'))
    probs,_=predict(key,experiment_name=config.EXPERIMENT,year=2018,
                      X=Xx[list(model.feature_names_in_)+['PATIENT_ID']], y=y)
    auc=roc_auc_score(probs.OBS,probs.PRED)
    recall, ppv, spec, newpred = performance(obs=probs.OBS, pred=probs.PRED, K=20000)
    
    brier=brier_score_loss(y_true=probs.OBS, y_prob=probs.PRED)
    ap=average_precision_score(probs.OBS,probs.PRED)
    table=pd.concat([table,
                     pd.DataFrame.from_dict({'Model':[key], 'AUC':[ auc], 'AP':[ap],
                      'R@20k': [recall], 'PPV@20K':[ppv], 
                      'Brier':[brier]})])
    probs['TOP20k']=newpred


#%%
test_logisticMetrics=pd.read_csv(config.METRICSPATH+'/metrics2018.csv')
test_logisticMetrics=test_logisticMetrics.loc[test_logisticMetrics.Algorithm=='logistic']
test_logisticMetrics.rename(columns={'Score':'AUC', 
                                     'Recall_20000':'R@20k', 'PPV_20000':'PPV@20K'},
                            inplace=True)
table=pd.concat([table,test_logisticMetrics[table.columns]])
print(table.round(4).to_latex(index=False))
