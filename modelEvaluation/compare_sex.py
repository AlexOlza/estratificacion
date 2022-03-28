#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:37 2022

@author: aolza
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
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
# sys.stdout = open(f'{config.ROOTPATH}estratificacion-reports/compare_sex_{config.EXPERIMENT}.md', 'w')
from modelEvaluation.compare import detect_models, compare, detect_latest, performance
import modelEvaluation.calibrate as cal
from dataManipulation.dataPreparation import getData
#%%
def plot_roc(fpr, tpr, groupname):
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                      estimator_name=groupname)
    # display.plot()
    
    return display

def plot_pr(precision, recall, groupname, y, pred):
    avg_prec = average_precision_score(y, pred)
    display = PrecisionRecallDisplay(precision=precision, recall=recall, 
                                     estimator_name=groupname,
                                     average_precision=avg_prec)
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
pastX,pasty=getData(year-2)
#%%
# X.drop('PATIENT_ID', axis=1, inplace=True)
predictors=X.columns
features=X.drop('PATIENT_ID', axis=1).columns

for column in features:
    if column!='FEMALE':
        X[f'{column}INTsex']=X[column]*X['FEMALE']
        pastX[f'{column}INTsex']=pastX[column]*pastX['FEMALE']
#%%
available_models=detect_models()
# latest=detect_latest(available_models)

female=X['FEMALE']==1
male=X['FEMALE']==0
sex=['Mujeres', 'Hombres']
#%%
import joblib as job
separate_cal,joint_cal,inter_cal={},{},{}
roc,roc_joint,roc_inter={k:{} for k in sex},{k:{} for k in sex},{k:{} for k in sex}
pr, pr_joint, pr_inter={k:{} for k in sex}, {k:{} for k in sex}, {k:{} for k in sex} #precision-recall curves
PRECISION, RECALL, THRESHOLDS={k:{} for k in sex}, {k:{} for k in sex}, {k:{} for k in sex}
FPR,TPR, ROCTHRESHOLDS={k:{} for k in sex}, {k:{} for k in sex}, {k:{} for k in sex}

table=pd.DataFrame()
K=20000
models=['Global', 'Separado', 'Interaccion']
fighist, (axhist,axhist2) = plt.subplots(1,2)

for i, group, groupname in zip([1,0],[female,male],sex):
    recall,ppv,spec, score, ap={},{},{},{},{}
    selected=[l for l in available_models if ((groupname in l ) or (bool(re.match('logistic\d+|logisticSexInteraction',l))))]
    print('Selected models: ',selected)
    # LOAD MODELS
    globalmodelname=list(set(selected)-set([f'logistic{groupname}'])-set(['logisticSexInteraction']))[0]
    separatemodelname=f'logistic{groupname}.joblib'
    globalmodel=job.load(config.MODELPATH+globalmodelname+'.joblib')
    interactionmodel=job.load(config.MODELPATH+'logisticSexInteraction.joblib')
    separatemodel=job.load(config.MODELPATH+separatemodelname)
    
    # SUBSET DATA
    Xgroup=X.loc[group]
    ygroup=y.loc[group]

    pastXgroup=pastX.loc[pastX['FEMALE']==i]
    pastygroup=pasty.loc[pastX['FEMALE']==i]

    assert (all(Xgroup['FEMALE']==1) or all(Xgroup['FEMALE']==0))
    
    pos=sum(np.where(ygroup[config.COLUMNS]>=1,1,0))
    pastpos=sum(np.where(pastygroup[config.COLUMNS]>=1,1,0))
    print('Sample size 2017',len(Xgroup), 'positive: ',pastpos, 'prevalence=', pastpos/len(pastXgroup))  
    print('Sample size 2018',len(pastXgroup), 'positive: ',pos, 'prevalence=', pos/len(Xgroup))  

    # PREDICT
    separate_cal[groupname]=cal.calibrate(f'logistic{groupname}',year,
                              predictors=[p for p in predictors if not p=='FEMALE'],
                              presentX=Xgroup[predictors].drop('FEMALE', axis=1),
                              presentY=ygroup,
                              pastX=pastXgroup[predictors].drop('FEMALE', axis=1),
                              pastY=pastygroup)

    joint_cal[groupname]=cal.calibrate(globalmodelname,year,
                              filename=f'{globalmodelname}_{groupname}',
                              predictors=predictors,
                              presentX=Xgroup[predictors], presentY=ygroup,
                              pastX=pastXgroup[predictors], pastY=pastygroup)
    inter_cal[groupname]=cal.calibrate('logisticSexInteraction',year,
                              filename=f'logisticSexInteraction_{groupname}',
                              presentX=Xgroup, presentY=ygroup,
                              pastX=pastXgroup[pastXgroup.columns],
                              pastY=pastygroup,
                              predictors=config.PREDICTORREGEX+'|INTsex')
    inter_preds=inter_cal[groupname].PRED
    joint_preds=joint_cal[groupname].PRED
    separate_preds=separate_cal[groupname].PRED

    assert all(inter_cal[groupname].OBS==joint_cal[groupname].OBS)
    assert all(separate_cal[groupname].OBS==joint_cal[groupname].OBS)
    
    obs=np.where(inter_cal[groupname].OBS>=1,1,0)
    axhist.hist(separate_preds,bins=1000,label=groupname)
    axhist2.hist(joint_preds,bins=1000,label=groupname)
    
    # METRICS
    for model, preds in zip(models,[joint_preds, separate_preds, inter_preds]):
        prec, rec, thre = precision_recall_curve(obs, preds)
        fpr, tpr, rocthresholds = roc_curve(obs, preds)
        FPR[groupname][model]=fpr
        TPR[groupname][model]=tpr
        PRECISION[groupname][model]=prec
        RECALL[groupname][model]=rec
        THRESHOLDS[groupname][model]=thre
        ROCTHRESHOLDS[groupname][model]=rocthresholds
        # CURVES - PLOTS
        roc[groupname][model]=plot_roc(fpr, tpr, groupname)
        pr[groupname][model]=plot_pr(prec,rec, groupname, obs, preds)

        recallK,ppvK, specK=performance(preds, obs, K)
        rocauc=roc_auc_score(obs, preds)
        avg_prec = average_precision_score(obs, preds)
        recall[model]=recallK
        spec[model]=specK
        score[model]=rocauc
        ppv[model]=ppvK
        ap[model]=avg_prec

    df=pd.DataFrame(list(zip([f'{model}- {groupname}' for model in models],
                             [score[model] for model in models],
                             [ap[model] for model in models],
                             [recall[model] for model in models],
                             [spec[model] for model in models],
                             [ppv[model] for model in models])),
                    columns=['Model', 'AUC', 'AP', f'Recall_{K}',f'Specificity_{K}', f'PPV_{K}'])
    table=pd.concat([df, table])
#%%
axhist.legend(prop={'size': 10})
axhist2.legend(prop={'size': 10})
fig1, (ax_sep1,ax_joint1, ax_inter1) = plt.subplots(1,3,figsize=(16,8))
fig2, (ax_sep2,ax_joint2, ax_inter2) = plt.subplots(1,3,figsize=(16,8))
for groupname in sex:
    for ax, model in zip((ax_sep1,ax_joint1, ax_inter1), models):
        roc[groupname][model].plot(ax)
        ax.set_title(model)
    for ax, model in zip((ax_sep2,ax_joint2, ax_inter2), models):
        pr[groupname][model].plot(ax)
        ax.set_title(model)

print(table.to_markdown(index=False,))
#%%
interFeatures=X.columns #preserves the order
globalFeatures=features #original columns (without interaction terms)
separateFeatures=[f for f in features if not f=='FEMALE'] #for the separate models

timesLargerForMen,ratio,m1,m2=beta_differences('logisticHombres.joblib', 'logisticMujeres.joblib',
                            separateFeatures)
oddsContribMuj={name:np.exp(value) for name, value in zip(separateFeatures, m2.coef_[0])}
oddsContribHom={name:np.exp(value) for name, value in zip(separateFeatures, m1.coef_[0])}
oddsContribGlobal={name:np.exp(value) for name, value in zip(globalFeatures, globalmodel.coef_[0])}
oddsContribInter={name:np.exp(value) for name, value in zip(interFeatures, interactionmodel.coef_[0])}
print('Global contrib of FEMALE variable is: ', oddsContribGlobal['FEMALE'])

N=5
print(f'TOP {N} positive and negative coefficients for women')
print(top_K_dict(oddsContribMuj, N))
print(top_K_dict(oddsContribMuj, N, reverse=False))
print('  ')
print(f'TOP {N} positive and negative coefficients for men')
print(top_K_dict(oddsContribHom, N))
print(top_K_dict(oddsContribHom, N, reverse=False))
print('-----'*5)
print('  ')
print(f'TOP {N} variables that increase the ODDS for men to be hospitalized more prominently than for women: ')
print(top_K_dict(timesLargerForMen, N))
print('  ')
print(f'TOP {N} variables that DECREASE the ODDS for men to be hospitalized more prominently than for women: ')
print(top_K_dict(timesLargerForMen, N, reverse=False))

oddsContrib={name:[muj, hom] for name, muj, hom in zip(features, oddsContribMuj.values(), oddsContribHom.values())}
oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',columns=['Mujeres', 'Hombres'])
oddsContrib['Global']=pd.Series(oddsContribGlobal)
oddsContrib['Interaccion']=pd.Series(oddsContribInter)
femaleContrib=pd.Series([np.nan, np.nan , oddsContribGlobal['FEMALE'],oddsContribInter['FEMALE']],index=['Mujeres', 'Hombres', 'Global', 'Interaccion'],name='FEMALE')
oddsContrib.loc['FEMALE']=femaleContrib
Xhom=X.loc[male]
Xmuj=X.loc[female]
oddsContrib['NMuj']=[Xmuj[name].sum() for name in oddsContrib.index]
oddsContrib['NHom']=[Xhom[name].sum() for name in oddsContrib.index]



oddsContrib.to_csv(config.MODELPATH+'sexSpecificOddsContributions.csv')
#%%
for i,model in enumerate(models):
    print(' ')
    print(model)
    print('what should the probability threshold be to get the same recall for women as for men? ')
    print('Recall for men (global model): ',table.Recall_20000.iloc[i])
    idx=[ n for n,i in enumerate(RECALL['Mujeres'][model]) if i<=table.Recall_20000.iloc[0]][0]
    t=THRESHOLDS['Mujeres'][model][idx]
    idx2=[ n for n,i in enumerate(ROCTHRESHOLDS['Mujeres'][model]) if i<=t][0]
    print('Threshold: ',t)
    print('Recall: ', RECALL['Mujeres'][model][idx])
    print('Number of selected women: ', sum(joint_preds>=t))

    print('Specificity for these women is: ',1-FPR['Mujeres'][model][idx2])

    print('And for men: ',table.Specificity_20000.iloc[i])
    print('PPV for these women is: ',PRECISION['Mujeres'][model][idx])
    print('ANd for men: ',table.PPV_20000.iloc[i])
    
#%% CALIBRATION CURVES
for title, preds in zip(['Global', 'Separado', 'Interaccion'], [joint_cal, separate_cal, inter_cal]):
    cal.plot(preds)
