#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:37 2022

@author: aolza
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay,roc_curve, auc
from modelEvaluation.predict import generate_filename
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')

import pandas as pd
#%%

config_used=input('Full path to configuration json file ')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)

from modelEvaluation.compare import detect_models, compare, detect_latest, performance
from dataManipulation.dataPreparation import getData
#%%
def plot_roc(y, pred, groupname):
    fpr, tpr, thresholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                      estimator_name=groupname)
    # display.plot()
    
    return display

def beta_differences(modelname1, modelname2, features):
    m1=job.load(config.MODELPATH+modelname1)
    m2=job.load(config.MODELPATH+modelname2)
    assert m1.n_features_in_==m2.n_features_in_, 'Models with different number of predictors!'
    assert m1.n_features_in_==len(features)
    diff=abs(m1.coef_-m2.coef_)
    ratio=m1.coef_/(m2.coef_+1e-14)
    diff={name:value for name, value in zip(features, diff[0])}
    ratio={name:value for name, value in zip(features, ratio[0])}
    return(diff, ratio,m1,m2)

def top_K_dict(d, K):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     return sorted(d.items(), key=lambda x: x[1], reverse=True)[:K]
#%%
year=int(input('YEAR TO PREDICT: ')) 
X,y=getData(year-1)
available_models=detect_models()
# latest=detect_latest(available_models)

female=X['FEMALE']==1
male=X['FEMALE']==0
sex=['Mujeres', 'Hombres']
#%%
import joblib as job
roc,roc_joint={},{}
table=pd.DataFrame()
K=20000
for group, groupname in zip([female,male],sex):
    selected=[l for l in available_models if ((groupname in l ) or (bool(re.match('logistic\d+',l))))]
    print(selected)
    globalmodel=list(set(selected)-set([f'logistic{groupname}']))[0]+'.joblib'
    separatemodel=f'logistic{groupname}.joblib'
    globalmodel=job.load(config.MODELPATH+globalmodel)
    separatemodel=job.load(config.MODELPATH+separatemodel)
    Xgroup=X.loc[group]
    ygroup=y.loc[group]
    assert (all(Xgroup['FEMALE']==1) or all(Xgroup['FEMALE']==0))
    # ygroup=np.where(ygroup[config.COLUMNS]>=1,1,0)
    # ygroup=ygroup.ravel()
    print('Sample size ',len(Xgroup), 'positive: ',sum(np.where(ygroup[config.COLUMNS]>=1,1,0)))  
    
    separate_preds=separatemodel.predict_proba(Xgroup.drop(['PATIENT_ID','FEMALE'],axis=1))[:,-1]
    joint_preds=globalmodel.predict_proba(Xgroup.drop(['PATIENT_ID'], axis=1))[:,-1]

    obs=np.where(ygroup[config.COLUMNS]>=1,1,0)
    
    roc[groupname]=plot_roc(obs,separate_preds, groupname)
    roc_joint[groupname]=plot_roc(obs,joint_preds, groupname)
    
    recall,ppv=performance(separate_preds, obs, K)
    score=roc_auc_score(obs, separate_preds)
    recalljoint,ppvjoint=performance(joint_preds, obs, K)
    scorejoint=roc_auc_score(obs, joint_preds)
    df=pd.DataFrame(list(zip([f'Global- {groupname}', f'Separado- {groupname}'],[scorejoint,score],[recalljoint,recall],[ppvjoint, ppv])),
                    columns=['Model', 'AUC', f'Recall_{K}', f'PPV_{K}'])
    table=pd.concat([df, table])
fig, (ax_sep,ax_joint) = plt.subplots(1,2,figsize=(10,8))
for groupname in sex:
    roc[groupname].plot(ax_sep)
    roc_joint[groupname].plot(ax_joint)
    
ax_sep.set_title('Separados')
ax_joint.set_title('Juntos')

print(table.to_markdown(index=False,))

diff,ratio,m1,m2=beta_differences('logisticHombres.joblib', 'logisticMujeres.joblib',
                            X.drop(['PATIENT_ID','FEMALE'],axis=1).columns)

print('TOP 5 coefficient differences are: ')
K=5
print(top_K_dict(diff, K))
print('TOP 5 coefficient ratios are: ')
print(top_K_dict(ratio, K))