#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:08:20 2023

@author: aolza

Modelo predictivo de riesgo de muerte a los 12 meses como elemento de ayuda para la identificación de personas con necesidades de cuidados paliativos

| Model                              |      AUC |       AP |    R@20k |   PPV@20K |      Brier |
|:-----------------------------------|---------:|---------:|---------:|----------:|-----------:|
| paliativos_canarias_Demo           | 0.911928 | 0.097352 | 0.177472 |   0.1447  | 0.0092119  |
| paliativos_canarias_DemoDiag       | 0.946049 | 0.2234   | 0.265473 |   0.2979  | 0.0086666  |
| paliativos_canarias_DemoDiagPharma | 0.948048 | 0.237277 | 0.27728  |   0.31115 | 0.00856241 |
"""

import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.death1year_canarias'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
import joblib
import pandas as pd
import numpy as np
from dataManipulation.dataPreparation import getData
from modelEvaluation.compare import performance
descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
model=joblib.load(config.MODELPATH+'paliativos_canarias_DemoDiagPharma.joblib')
preds=pd.read_csv(config.PREDPATH+'paliativos_canarias_DemoDiagPharma__2018.csv')
X, y = getData(2017)
#%%
future_dead=pd.read_csv(config.FUTUREDECEASEDFILE)
dead2019=future_dead.loc[future_dead.date_of_death.str.startswith('2019')].PATIENT_ID
preds['OBS2019']=np.where(preds.PATIENT_ID.isin(dead2019),1,0)
#%%
baseline_risk=np.exp(model.intercept_[0])
print('Probabilidad basal de fallecer: ',baseline_risk)

recall2019, ppv2019, _, _ = performance(obs=preds.OBS+preds.OBS2019, pred=preds.PRED, K=20000)
recall, ppv, spec, newpred= performance(obs=preds.OBS, pred=preds.PRED, K=20000)
preds['TopK']=newpred.astype(int)
#%%
coefs=pd.DataFrame.from_dict({k:[v] for v,k in zip(model.coef_[0],model.feature_names_in_)},orient='columns')
coefs.rename(columns={0:'beta'},inplace=True)
#%%


#%%
def explain_example(lower,upper, preds, coefs, X, descriptions, random_state):
    patient_A=preds.loc[preds.PRED>lower].loc[preds.PRED<upper].sample(1,random_state=random_state)
    patient_A_CCS=X.loc[X.PATIENT_ID==patient_A.PATIENT_ID.values[0]]
    patient_A_CCS=[c for c in patient_A_CCS if patient_A_CCS[c].values[0]==1]
    df_A=pd.concat([coefs[patient_A_CCS].T.nlargest(5,0),coefs[patient_A_CCS].T.nsmallest(5,0)])
    df_A['CATEGORIES']=df_A.index
    df_A=pd.merge(df_A,descriptions,on='CATEGORIES',how='left')
    return patient_A, df_A

patient_A, df_A=explain_example(0.95,0.96,preds.loc[preds.TopK==1],coefs,X,descriptions,0)
patient_B, df_B=explain_example(0.2,0.25,preds.loc[preds.TopK==1],coefs,X,descriptions,0)
#%%
