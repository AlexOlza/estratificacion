#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:44:04 2023

@author: aolza
"""
import sys
import os
import configurations.utility as util
from python_settings import settings as config
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()

logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import joblib
import pandas as pd
import re
import numpy as np

def concat_preds(file1,file2):
    muj=pd.read_csv(file1)
    muj['FEMALE']=1
    hom=pd.read_csv(file2)
    hom['FEMALE']=0
    return pd.concat([muj,hom])

CCS=eval(input('CCS? True/False: '))
ccs='CCS' if CCS else 'ACG'

if CCS:
    linear_modelpath=config.ROOTPATH+'models/costCCS_parsimonious/'
    linear_modelname='linear20230324_130625'
    linear_predpath=re.sub('models','predictions',linear_modelpath)
else: #ACG
    linear_modelpath=config.ROOTPATH+'models/cost_ACG/'
    linear_modelname='linear20221018_103900'
    linear_predpath=re.sub('models','predictions',linear_modelpath)
#%%
allpreds=concat_preds(linear_predpath+f'{linear_modelname}_Mujeres__2018.csv',
                      linear_predpath+f'{linear_modelname}_Hombres__2018.csv') 
allpreds.loc[allpreds.PRED<0,'PRED']=0

allpreds['PredTop20k']=np.where(allpreds.PATIENT_ID.isin(allpreds.nlargest(20000,'PRED').PATIENT_ID),1,0)
allpreds['ObsTop20k']=np.where(allpreds.PATIENT_ID.isin(allpreds.nlargest(20000,'OBS').PATIENT_ID),1,0)

N=int(0.01*len(allpreds))
for i in [1,2,3,4,5,10,20,25]:
    allpreds[f'ObsTop{i}%']=np.where(allpreds.PATIENT_ID.isin(allpreds.nlargest(i*N,'OBS').PATIENT_ID),1,0)

#%%
from sklearn.metrics import precision_score, recall_score, 
PPV={}
PPV['Top20k']=100*precision_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop20k'])
for i in [1,2,3,4,5,10,20,25]:
    PPV[N*i]=100*precision_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop{i}%'])
#%%
PPV=pd.DataFrame.from_dict(PPV, orient='index',columns=['PPV'])
PPV['N']=PPV.index
PPV.index=[round(20000/N,2),1,2,3,4,5,10,20,25]
PPV['Percentile']=PPV.index
print(PPV.to_markdown(index=False))
#%%
SENS={}
SENS['Top20k']=100*recall_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop20k'])
for i in [1,2,3,4,5,10,20,25]:
    SENS[N*i]=100*recall_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop{i}%'])
#%%
SENS=pd.DataFrame.from_dict(SENS, orient='index',columns=['SENS'])
SENS['N']=SENS.index
SENS.index=[round(20000/N,2),1,2,3,4,5,10,20,25]
SENS['Percentile']=SENS.index
#%%
SPEC={}
SPEC['Top20k']=100*recall_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop20k'],pos_label=0)
for i in [1,2,3,4,5,10,20,25]:
    SPEC[N*i]=100*recall_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop{i}%'],pos_label=0)
SPEC=pd.DataFrame.from_dict(SPEC, orient='index',columns=['SPEC'])
SPEC['N']=SPEC.index
SPEC.index=[round(20000/N,2),1,2,3,4,5,10,20,25]
SPEC['Percentile']=SPEC.index
#%%
NPV={}
NPV['Top20k']=100*precision_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop20k'],pos_label=0)
for i in [1,2,3,4,5,10,20,25]:
    NPV[N*i]=100*precision_score(y_pred=allpreds.PredTop20k, y_true=allpreds[f'ObsTop{i}%'],pos_label=0)
NPV=pd.DataFrame.from_dict(NPV, orient='index',columns=['NPV'])
NPV['N']=NPV.index
NPV.index=[round(20000/N,2),1,2,3,4,5,10,20,25]
NPV['Percentile']=NPV.index
#%%
print(SENS.merge(PPV,on=['N','Percentile']).merge(SPEC,on=['N','Percentile']).merge(NPV,on=['N','Percentile'])[['Percentile','N','SENS','PPV', 'NPV', 'SPEC']].to_markdown(index=False))
