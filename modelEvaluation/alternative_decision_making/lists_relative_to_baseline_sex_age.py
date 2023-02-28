#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:21:35 2023

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster


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
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import pandas as pd
import re
import numpy as np
import joblib as job
import matplotlib.pyplot as plt
from modelEvaluation.predict import predict
from modelEvaluation.detect import detect_models, detect_latest
from sklearn.metrics import confusion_matrix
import re
#%%
def performance(obs, pred, K, computemetrics=True, verbose=True):
    assert K>0, 'K must be greater than zero'
    assert len(obs)==len(pred), 'obs and pred must have the same length'
    if K<1: #K can be expressed in proportion form, for example 0.01 to use 1% of the population
        K_=K
        K=round(len(obs)*K)
        print(f'The top {100*K_}% list has {K} patients')
    
    
    
    orderedPred = sorted(pred, reverse=True)   
    cutoff = orderedPred[K - 1]

    newpred = (pred >= cutoff).astype(int)

    if 'COSTE_TOTAL_ANO2' in config.COLUMNS:  # maybe better: not all([int(i)==i for i in obs])
        orderedObs = sorted(obs, reverse=True)
        newobs = obs >= orderedObs[K - 1]
    else:
        newobs = np.where(obs >= 1, 1, 0)  # Whether the patient had ANY admission
    c = confusion_matrix(y_true=newobs, y_pred=newpred)
    if verbose: print(c)
    tn, fp, fn, tp = c.ravel()
    if not computemetrics:
        return (tn, fp, fn, tp)
    if verbose: print(' tn, fp, fn, tp =', tn, fp, fn, tp)
    recall = c[1][1] / (c[1][0] + c[1][1])
    ppv = c[1][1] / (c[0][1] + c[1][1])
    specificity = tn / (tn + fp)
    if verbose: print('Recall, PPV, Spec = ', recall, ppv, specificity)
    return (recall, ppv, specificity, newpred)

def healthy_preds_age(model,X):
    if not 'COSTE_TOTAL_ANO2' in config.COLUMNS:
        def predict_fn(x): return model.predict_proba(x)[:,1]  # Whether the patient had ANY admission
    else:
        def predict_fn(x): return model.predict(x)[0]
    zeros=pd.DataFrame([np.zeros(X.shape[1]-1)],columns=X.drop('PATIENT_ID',axis=1).columns)
    healthy_patients={}
    healthy_predictions={}
    agecols=[c for c in X if 'AGE' in c]
    for c in agecols:
        healthy_patients[f'FEMALE=1{c}']=zeros.copy()
        healthy_patients[f'FEMALE=1{c}'][['FEMALE',c]]=1
        
        healthy_patients[f'FEMALE=0{c}']=zeros.copy()
        healthy_patients[f'FEMALE=0{c}'][[c]]=1
        
        healthy_predictions[f'FEMALE=1{c}']=predict_fn(healthy_patients[f'FEMALE=1{c}'])
        healthy_predictions[f'FEMALE=0{c}']=predict_fn(healthy_patients[f'FEMALE=0{c}'])

    healthy_patients[f'FEMALE=1AGE_85GT']=zeros.copy()
    healthy_patients[f'FEMALE=1AGE_85GT']['FEMALE']=1
    healthy_predictions['FEMALE=0AGE_85GT']=predict_fn(zeros.copy())
    healthy_predictions['FEMALE=1AGE_85GT']=predict_fn(healthy_patients[f'FEMALE=1AGE_85GT'])
    return healthy_predictions

def healthy_preds_no_age(model,X):
    if not 'COSTE_TOTAL_ANO2' in config.COLUMNS:
        def predict_fn(x): return model.predict_proba(x)[:,1]  # Whether the patient had ANY admission
    else:
        def predict_fn(x): return model.predict(x)[0]
    zeros=pd.DataFrame([np.zeros(X.shape[1]-1)],columns=X.drop('PATIENT_ID',axis=1).columns)
    healthy_patients={}
    healthy_predictions={}
    healthy_patients[f'FEMALE=1AGE_85GT']=zeros.copy()
    healthy_patients[f'FEMALE=1AGE_85GT']['FEMALE']=1
    healthy_predictions['FEMALE=0AGE_85GT']=predict_fn(zeros.copy())
    healthy_predictions['FEMALE=1AGE_85GT']=predict_fn(healthy_patients[f'FEMALE=1AGE_85GT'])
    return healthy_predictions
#%%
year=2017#eval(input('Year: '))
X, y = getData(year)
available_models = detect_models()


#%%
selected_modelname=[l for l in detect_latest(available_models) if (bool(re.match(f'{config.ALGORITHM}\d+', l)))][0]
model=job.load(config.MODELPATH+f'{selected_modelname}.joblib')

if not 'AGE_85GT' in X:
    X['AGE_85GT']=np.where(X.filter(regex=("AGE*")).sum(axis=1)==0,1,0)
    
X['AGE']=X.filter(regex='AGE_',axis=1).idxmax(1)
XageSex=X[['PATIENT_ID','AGE','FEMALE']]

X.drop(['AGE','AGE_85GT'],axis=1,inplace=True)

preds,score=predict(selected_modelname,config.EXPERIMENT,2018)
preds=pd.merge(preds,XageSex,on='PATIENT_ID')

healthy=healthy_preds_age(model, X)
healthy_no_age=healthy_preds_no_age(model, X)
#%%
preds['BASELINE']=0
preds['BASELINE_WITH_AGE']=0
for sex in preds.FEMALE.unique():
    preds.loc[(preds.FEMALE==sex),'BASELINE']=healthy_no_age[f'FEMALE={sex}AGE_85GT'][0]
    for age in preds.AGE.unique():
        key=f'FEMALE={sex}{age}'
        preds.loc[(preds.FEMALE==sex) & (preds.AGE==age),'BASELINE_WITH_AGE']=healthy[key][0]
#%%
perf={}
perf['global']=performance(preds.OBS, preds.PRED/preds.BASELINE, K=20000)
#%% women and men separately
from matplotlib import pyplot as plt
import seaborn as sns


fig,ax=plt.subplots()
sns.set_style('whitegrid')
for sex in preds.FEMALE.unique():
    p=preds.loc[preds.FEMALE==sex]
    print((p.PRED/p.BASELINE).describe())
    perf[sex]=performance(p.OBS, p.PRED/p.BASELINE, K=20000)
    sns.kdeplot(p.PRED/p.BASELINE,ax=ax,label=sex)

#%%
females=preds.loc[preds.FEMALE==1]
males=preds.loc[preds.FEMALE==0]
top_10k_and_10k=pd.concat([females.nlargest(10000,'PRED'),
          males.nlargest(10000,'PRED')])
preds['top_10k_and_10k']=np.where(preds.PATIENT_ID.isin(top_10k_and_10k.PATIENT_ID), 1,0)

preds['relative_sex']=preds.PRED/preds.BASELINE
preds['top_20k_relative_sex']=np.where(
    preds.PATIENT_ID.isin(preds.nlargest(20000,'relative_sex').PATIENT_ID),1,0)

preds['relative_sex_age']=preds.PRED/preds.BASELINE_WITH_AGE
preds['top_20k_relative_sex_age']=np.where(
    preds.PATIENT_ID.isin(preds.nlargest(20000,'relative_sex_age').PATIENT_ID),1,0)

preds['top_20k']=np.where(
    preds.PATIENT_ID.isin(preds.nlargest(20000,'PRED').PATIENT_ID),1,0)

print('Top 20k % women: ',preds.loc[preds.FEMALE==1]['top_20k'].sum()*100/20000)
print('Top 10k and 10k % women: ',preds.loc[preds.FEMALE==1]['top_10k_and_10k'].sum()*100/20000)
print('Top 20k relative sex % women: ',preds.loc[preds.FEMALE==1]['top_20k_relative_sex'].sum()*100/20000)
print('Top 20k relative sex and age % women: ',preds.loc[preds.FEMALE==1]['top_20k_relative_sex_age'].sum()*100/20000)
#%%

#%%
def table(preds,cols=['top_20k', 'top_10k_and_10k','top_20k_relative_sex','top_20k_relative_sex_age'],K=20000):
    df=pd.DataFrame()
    perf={}
    
    for col in cols:
        if not 'COSTE_TOTAL_ANO2' in config.COLUMNS:
            newobs = np.where(preds.OBS >= 1, 1, 0)  # Whether the patient had ANY admission
            from sklearn.metrics import roc_auc_score as score
            obs=newobs
        else:
            from sklearn.metrics import r2_score as score
            newobs= np.where(preds.OBS.isin(preds.OBS.nlargest(K)), 1, 0)
            obs=preds.OBS
        c = confusion_matrix(y_true=newobs, y_pred=preds[col])
        recall = c[1][1] / (c[1][0] + c[1][1])
        ppv = c[1][1] / (c[0][1] + c[1][1])
        perf['global']=[score(obs, preds.PRED),recall,ppv]
        for sex in preds.FEMALE.unique():
            p=preds.loc[preds.FEMALE==sex]
            if not 'COSTE_TOTAL_ANO2' in config.COLUMNS:
                newobs = np.where(p.OBS >= 1, 1, 0)  # Whether the patient had ANY admission
                obs=newobs
            else:
                newobs= np.where(p.OBS.isin(p.OBS.nlargest(K)), 1, 0) 
                obs=p.OBS
            c = confusion_matrix(y_true=newobs, y_pred=p[col])
            recall = c[1][1] / (c[1][0] + c[1][1])
            ppv = c[1][1] / (c[0][1] + c[1][1])
            perf[sex]=[score(obs, p.PRED),recall,ppv]
        df_=pd.DataFrame(perf).rename(columns={0:'Men',1:'Women'},
                                     index={0:f'Score_{col}',1:f'Recall_{col}',2:f'PPV_{col}'})
        df=pd.concat([df,df_])
    df['Ratio']=df.Women/df.Men
    return df

#%%
df=table(preds)


"""

urgcmsCCS_parsimonious

Top 20k % women:  38.06
Top 20k relative sex % women:  53.14
Top 20k relative sex and age % women:  50.925

                                   global       Men     Women     Ratio
Score_top_20k                    0.804270  0.807296  0.801028  0.992237
Recall_top_20k                   0.076966  0.095282  0.057549  0.603989
PPV_top_20k                      0.490700  0.504682  0.467945  0.927208
Score_top_20k_relative_sex       0.804270  0.807296  0.801028  0.992237
Recall_top_20k_relative_sex      0.074903  0.074906  0.074901  0.999940
PPV_top_20k_relative_sex         0.477550  0.524434  0.436206  0.831765
Score_top_20k_relative_sex_age   0.804270  0.807296  0.801028  0.992237
Recall_top_20k_relative_sex_age  0.071892  0.075347  0.068228  0.905517
PPV_top_20k_relative_sex_age     0.458350  0.503719  0.414629  0.823137

costCCS_parsimonious
                                   global       Men     Women     Ratio
Score_top_20k                    0.212348  0.239607  0.181984  0.759512
Recall_top_20k                   0.221950  0.206350  0.138350  0.670463
PPV_top_20k                      0.221950  0.340962  0.350431  1.027771
Score_top_20k_relative_sex       0.212348  0.239607  0.181984  0.759512
Recall_top_20k_relative_sex      0.218150  0.173850  0.170550  0.981018
PPV_top_20k_relative_sex         0.218150  0.369461  0.322127  0.871882
Score_top_20k_relative_sex_age   0.212348  0.239607  0.181984  0.759512
Recall_top_20k_relative_sex_age  0.164250  0.128150  0.145350  1.134218
PPV_top_20k_relative_sex_age     0.164250  0.317439  0.243753  0.767875"""
#RR = OR/[(1-probability in reference group) + (probability in reference group x OR)]