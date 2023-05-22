#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess the predictive performance gain
of the ACG variable, task #48 
Created on Mon Feb  7 12:30:42 2022

@author: alex
What should I do with these? Ask Edu
HOSDOM|FRAILTY|INGRED_14GT
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
X,y=getData(2016)#new data

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

variables={f'{prefix}Demo':'PATIENT_ID|FEMALE|AGE_[0-9]+$',
           f'{prefix}DemoDiag':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_' if 'ACG' in config.PREDICTORREGEX else 'PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS',
           f'{prefix}DemoDiagPharma':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_|RXMG_' if 'ACG' in config.PREDICTORREGEX else CCSPHARMA,
            f'{prefix}DemoDiagPharmaIsomorb':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_(?!NUR11|RES10)|RXMG_(?!ZZZX000)|ACG_' if 'ACG' in config.PREDICTORREGEX else CCSGMA
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
X=X[[c for c in X if X[c].max()>0]]
PATIENT_ID=X.PATIENT_ID
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
   
y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y
#%%
# future_dead=pd.read_csv(config.FUTUREDECEASEDFILE)
# dead2019=future_dead.loc[future_dead.date_of_death.str.startswith('2019')].PATIENT_ID
import joblib
print('JOBLIB VERSION',joblib.__version__)
table=pd.DataFrame()
for key, val in variables.items():
    Xx=X.copy()
    if not val:
        continue
    try:
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
            # false_positives=probs.loc[(probs.OBS.values==0) ] #are not actually positives!!
        # n_selected=len(false_positives) # this is the length of list, should be K with enough flexibility (not for Demo)
        # false_positives=false_positives.loc[(false_positives.TOP20k.values==1)]
        # almost_true=len(false_positives.loc[false_positives.PATIENT_ID.isin(dead2019)])
        # print(f'Out of {len(false_positives)} false positives, {almost_true} died in 2019 ({round(almost_true*100/len(false_positives),2)}%)')
    except:
        print(key , 'Failed')

#%%
print(table.to_markdown(index=False))