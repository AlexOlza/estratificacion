#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:37 2022

@author: aolza
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
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

def plot_pr(y, pred, groupname):
    precision, recall, _ = precision_recall_curve(y, pred)
    display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=groupname)
    return display
def beta_differences(modelname1, modelname2, features):
    m1=job.load(config.MODELPATH+modelname1)
    m2=job.load(config.MODELPATH+modelname2)
    assert m1.n_features_in_==m2.n_features_in_, 'Models with different number of predictors!'
    assert m1.n_features_in_==len(features)
    diff=m1.coef_-m2.coef_
    ratio=m1.coef_/(m2.coef_+1e-14)
    timesLargerForMen={name:np.exp(value) for name, value in zip(features, diff[0])}
    ratio={name:value for name, value in zip(features, ratio[0])}
    return(timesLargerForMen, ratio,m1,m2)

def top_K_dict(d, K, reverse=True):
    """ a) create a list of the dict's keys and values; 
     b) return the key with the max value"""  
    items=sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    if all([v>=0 for v in d.values()]):
        return items[:K]
    else:
        return items[:K]+items[-K:]
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
pr, pr_joint={}, {} #precision-recall curves
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
    pr_joint[groupname]=plot_pr(obs, joint_preds, groupname)
    pr[groupname]=plot_pr(obs, separate_preds, groupname)
    
    recall,ppv=performance(separate_preds, obs, K)
    score=roc_auc_score(obs, separate_preds)
    recalljoint,ppvjoint=performance(joint_preds, obs, K)
    scorejoint=roc_auc_score(obs, joint_preds)
    df=pd.DataFrame(list(zip([f'Global- {groupname}', f'Separado- {groupname}'],[scorejoint,score],[recalljoint,recall],[ppvjoint, ppv])),
                    columns=['Model', 'AUC', f'Recall_{K}', f'PPV_{K}'])
    table=pd.concat([df, table])
fig1, (ax_sep1,ax_joint1) = plt.subplots(1,2,figsize=(10,8))
fig2, (ax_sep2,ax_joint2) = plt.subplots(1,2,figsize=(10,8))
for groupname in sex:
    roc[groupname].plot(ax_sep1)
    roc_joint[groupname].plot(ax_joint1)
    pr[groupname].plot(ax_sep2)
    pr_joint[groupname].plot(ax_joint2)

ax_sep1.set_title('Separados')
ax_joint1.set_title('Juntos')
ax_sep2.set_title('Separados')
ax_joint2.set_title('Juntos')

print(table.to_markdown(index=False,))
features=X.drop(['PATIENT_ID','FEMALE'],axis=1).columns
globalFeatures=X.drop(['PATIENT_ID'],axis=1).columns #preserves the order
timesLargerForMen,ratio,m1,m2=beta_differences('logisticHombres.joblib', 'logisticMujeres.joblib',
                            features)
oddsContribMuj={name:np.exp(value) for name, value in zip(features, m2.coef_[0])}
oddsContribHom={name:np.exp(value) for name, value in zip(features, m1.coef_[0])}
oddsContribGlobal={name:np.exp(value) for name, value in zip(globalFeatures, globalmodel.coef_[0])}
print('Global contrib of FEMALE variable is: ', oddsContribGlobal['FEMALE'])

K=5
print(f'TOP {K} positive and negative coefficients for women')
print(top_K_dict(oddsContribMuj, K))
print(top_K_dict(oddsContribMuj, K, reverse=False))
print('  ')
print(f'TOP {K} positive and negative coefficients for men')
print(top_K_dict(oddsContribHom, K))
print(top_K_dict(oddsContribHom, K, reverse=False))
print('-----'*5)
print('  ')
print(f'TOP {K} variables that increase the ODDS for men to be hospitalized more prominently than for women: ')
print(top_K_dict(timesLargerForMen, K))
print('  ')
print(f'TOP {K} variables that DECREASE the ODDS for men to be hospitalized more prominently than for women: ')
print(top_K_dict(timesLargerForMen, K, reverse=False))

oddsContrib={name:[muj, hom] for name, muj, hom in zip(features, oddsContribMuj.values(), oddsContribHom.values())}
oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',columns=['Mujeres', 'Hombres'])
oddsContrib['Global']=pd.Series(oddsContribGlobal)
femaleContrib=pd.Series([np.nan, np.nan , oddsContribGlobal['FEMALE']],index=['Mujeres', 'Hombres', 'Global'],name='FEMALE')
oddsContrib.loc['FEMALE']=femaleContrib
Xhom=X.loc[male]
Xmuj=X.loc[female]
oddsContrib['NMuj']=[Xmuj[name].sum() for name in oddsContrib.index]
oddsContrib['NHom']=[Xhom[name].sum() for name in oddsContrib.index]



oddsContrib.to_csv(config.MODELPATH+'sexSpecificOddsContributions.csv')
