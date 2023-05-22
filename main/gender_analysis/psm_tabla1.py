#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

a reference: https://sci-hub.se/10.1213/ANE.0000000000002787
Created on Tue Nov  8 10:32:34 2022

@author: alex
Source: https://github.com/konosp/propensity-score-matching/blob/main/propensity_score_matching_v2.ipynb
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
import joblib as job
from dataManipulation.dataPreparation import getData
from modelEvaluation.compare import performance
import numpy as np
import os
from time import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
#%%

def load_pairs(filename,directory=config.DATAPATH):
    pairs=pd.DataFrame()
    t0=time()
    print('Loading ',filename)
    for chunk in pd.read_csv(filename, chunksize=100000):
        d = dict.fromkeys(chunk.columns, np.int8)
        d['PATIENT_ID']=np.int64
        d['counterfactual']=np.int64
        d['propensity_score']=np.float64
        d['propensity_score_logit']=np.float64
        chunk= chunk.astype(d)
        pairs = pd.concat([pairs, chunk], ignore_index=True)

    util.vprint('Loaded in ',time()-t0,' seconds')
    return(pairs)
#%%
np.random.seed(config.SEED)
X,y=getData(2016)
original_columns=X.columns
print('Sample size ',len(X), 'females: ',X.FEMALE.sum())
assert not 'AGE_85GT' in X.columns

#%%
filename=os.path.join(config.DATAPATH,'single_neighbour_pairs.csv')
ps_model_filename=os.path.join(config.MODELPATH,'logistic_propensity_score_model.joblib')
#%%
pairs=load_pairs(filename)
#%%
females_=pairs.loc[pairs.FEMALE==1].drop_duplicates('counterfactual')
males_=pairs.loc[pairs.FEMALE==0].drop_duplicates(original_columns)
pairs=pd.concat([males_, females_]).sample(frac=1)# to shuffle the data
pairs.index=range(len(pairs))

pairs['outcome']=pd.merge(pairs[['PATIENT_ID']],y,on='PATIENT_ID')[config.COLUMNS]
#%%
threshold= 0 if config.ALGORITHM=='logistic' else y[config.COLUMNS].mean().values[0]
X['binary_outcome']=(y[config.COLUMNS]>threshold)
pairs['binary_outcome']=(pairs.outcome>threshold)

#%%
overview = pairs[['outcome','FEMALE']].groupby(by = ['FEMALE']).aggregate([np.mean, np.var, np.std, 'count'])
print(overview)

treated_outcome = overview['outcome']['mean'][1]
treated_counterfactual_outcome = overview['outcome']['mean'][0]
att = treated_outcome - treated_counterfactual_outcome
print('The Average Treatment Effect in females is (ATT): {:.4f}'.format(att))
#Vemos que, en una muestra de mujeres y hombres con las mismas 
#características clínicas, las mujeres ingresan menos (ATT<0)
#%%
#%%
overview2 = pairs[['binary_outcome','FEMALE']].groupby(by = ['FEMALE']).aggregate([np.mean, np.var, np.std, 'count'])
print(overview2)
treated_outcome = overview2['binary_outcome']['mean'][1]
treated_counterfactual_outcome = overview2['binary_outcome']['mean'][0]
att2 = treated_outcome - treated_counterfactual_outcome
print('For presence/absence of outcome, the Average Treatment Effect in females is (ATT): {:.4f}'.format(att2))

#%%
table1=pd.DataFrame()
for i,df in X.groupby('FEMALE'):
    table1[f'BEFORE -FEMALE={i}']=df[original_columns].drop(['PATIENT_ID','FEMALE'], axis=1).mean()
for i,df in pairs.groupby('FEMALE'):
    table1[f'AFTER -FEMALE={i}']=df[original_columns].drop(['PATIENT_ID','FEMALE'], axis=1).mean()
table1['Variable']=table1.index
#%%
#Build dataframe with descriptions
ccs=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str, usecols=['CCS CATEGORY', 'CCS CATEGORY DESCRIPTION'])
ccs.drop_duplicates(inplace=True)
ccs['CCS CATEGORY']=[f'CCS{i}' for i in ccs['CCS CATEGORY'].values]
#%%
table1=pd.merge(table1,ccs, left_on='Variable',right_on='CCS CATEGORY', how='left')
table1.round(3)[['Variable','BEFORE -FEMALE=0', 'BEFORE -FEMALE=1', 'AFTER -FEMALE=0',
       'AFTER -FEMALE=1', 'CCS CATEGORY DESCRIPTION']].to_csv(os.path.join(config.DATAPATH,'tabla1_promedios_psm.csv'), index=False)

#%%
table1=pd.DataFrame()
for i,df in X.groupby('FEMALE'):
    df=(df>0).astype(int)
    df=df[original_columns].drop(['PATIENT_ID','FEMALE'], axis=1)
    table1[f'BEFORE -FEMALE={i}']=100*df.sum()/df.count()
for i,df in pairs.groupby('FEMALE'):
    df=(df>0).astype(int)
    df=df[original_columns].drop(['PATIENT_ID','FEMALE'], axis=1)
    table1[f'AFTER -FEMALE={i}']=100*df.sum()/df.count()
table1['Variable']=table1.index

#%%
table1=pd.merge(table1,ccs, left_on='Variable',right_on='CCS CATEGORY', how='left')
table1.round(4)[['Variable','BEFORE -FEMALE=0', 'BEFORE -FEMALE=1', 'AFTER -FEMALE=0',
       'AFTER -FEMALE=1', 'CCS CATEGORY DESCRIPTION']].to_csv(os.path.join(config.DATAPATH,'tabla1_porcentajes_presencia_ausencia_psm.csv'), index=False)
