#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:37 2022

@author: aolza
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
from modelEvaluation.predict import generate_filename
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd
#%%
from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    config_used=input('Full path to configuration json file ')
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

def translateVariables(df,**kwargs):
     dictionaryFile=kwargs.get('file',os.path.join(config.INDISPENSABLEDATAPATH+'diccionarioACG.csv'))
     dictionary=pd.read_csv(dictionaryFile)
     dictionary.index=dictionary.codigo
     df=pd.merge(df, dictionary, right_index=True, left_index=True)
     return df
 
def beta_std_error(logModel, X, ):
    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = logModel.predict_proba(X)
    
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    V = np.diagflat(np.product(predProbs, axis=1))
    
    # Covariance matrix
    # Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
    # Python 3.5+, you can replace the covLogit-line below with the more readable:
    covLogit = np.linalg.inv(X_design.T @ V @ X_design)
    # Standard errors
    stderr=np.sqrt(np.diag(covLogit))
    print("Standard errors: ", stderr)
    # Wald statistic (coefficient / s.e.) ^ 2
    logitParams = np.insert(resLogit.coef_, 0, resLogit.intercept_)
    print("Wald statistics: ", (logitParams / np.sqrt(np.diag(covLogit))) ** 2)
    return stderr
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

        recallK,ppvK, specK, _=performance(preds, obs, K)
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
fig1, (ax_sep1,ax_joint1, ax_inter1) = plt.subplots(1,3,figsize=(16,8))
fig2, (ax_sep2,ax_joint2, ax_inter2) = plt.subplots(1,3,figsize=(16,8))
for groupname in sex:
    for ax, model in zip((ax_sep1,ax_joint1, ax_inter1), models):
        roc[groupname][model].plot(ax)
        ax.set_title(model)
    for ax, model in zip((ax_sep2,ax_joint2, ax_inter2), models):
        pr[groupname][model].plot(ax)
        ax.set_title(model)
fig1.savefig(os.path.join(config.FIGUREPATH,'rocCurve.png'))
fig2.savefig(os.path.join(config.FIGUREPATH,'prCurve.png'))
plt.show()
print(table.to_markdown(index=False,))
#%%
interFeatures=X.columns #preserves the order
globalFeatures=features #original columns (without interaction terms)
separateFeatures=[f for f in features if not f=='FEMALE'] #for the separate models

modeloH=job.load(config.MODELPATH+'logisticHombres.joblib')
modeloM=job.load(config.MODELPATH+'logisticMujeres.joblib')

oddsContribMuj={name:np.exp(value) for name, value in zip(separateFeatures, modeloM.coef_[0])}
oddsContribHom={name:np.exp(value) for name, value in zip(separateFeatures, modeloH.coef_[0])}
oddsContribGlobal={name:np.exp(value) for name, value in zip(globalFeatures, globalmodel.coef_[0])}
oddsContribInter={name:np.exp(value) for name, value in zip(interFeatures, interactionmodel.coef_[0])}
print('Global contrib of FEMALE variable is: ', oddsContribGlobal['FEMALE'])

oddsContrib={name:[muj, hom] for name, muj, hom in zip(separateFeatures, oddsContribMuj.values(), oddsContribHom.values())}
oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',columns=['Mujeres', 'Hombres'])
oddsContrib['Global']=pd.Series(oddsContribGlobal)
oddsContrib['Interaccion']=pd.Series(oddsContribInter)
femaleContrib=pd.Series([np.nan, np.nan , oddsContribGlobal['FEMALE'],oddsContribInter['FEMALE']],index=['Mujeres', 'Hombres', 'Global', 'Interaccion'],name='FEMALE')
oddsContrib.loc['FEMALE']=femaleContrib
oddsContrib['ratioH/M']=oddsContrib['Hombres']/oddsContrib['Mujeres']
oddsContrib['ratioM/H']=1/oddsContrib['ratioH/M']
Xhom=X.loc[male]
Xmuj=X.loc[female]
oddsContrib['NMuj']=[Xmuj[name].sum() for name in oddsContrib.index]
oddsContrib['NHom']=[Xhom[name].sum() for name in oddsContrib.index]

oddsContrib=translateVariables(oddsContrib)

#%% PRINT RISK FACTORS
N=5
for s, col in zip(['Mujeres', 'Hombres'], ['NMuj', 'NHom']):
    print(f'TOP {N} factores de riesgo para {s}')
    print(oddsContrib.sort_values(by=s, ascending=False).head(N)[[s,col,'descripcion']])
    print('  ')
    print(f'TOP {N} factores protectores para {s}')
    print(oddsContrib.sort_values(by=s, ascending=True).head(N)[[s, col,'descripcion']])
    print('  ')
print('-----'*N)
print('  ')

print(f'TOP {N} variables cuya presencia acrecienta el riesgo para los hombres más que para las mujeres: ')
print(oddsContrib.sort_values(by='ratioH/M', ascending=False).head(N)[['ratioH/M','NMuj', 'NHom','descripcion']])
print('  ')
print(f'TOP {N} variables cuya presencia acrecienta el riesgo para las mujeres más que para los hombres: ')
print(oddsContrib.sort_values(by='ratioM/H', ascending=False).head(N)[['ratioM/H','NMuj', 'NHom','descripcion']])

#%%
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
    
# %% CALIBRATION CURVES
# for title, preds in zip(['Global', 'Separado', 'Interaccion'], [joint_cal, separate_cal, inter_cal]):
#     cal.plot(preds,filename=title)

#%% 
"""
Si empleamos el modelo global y seleccionamos a los 20000 de mayor riesgo
(independientemente de si son hombres o mujeres) 
¿Cual sería el número de hombres y mujeres seleccionados? 
¿Cual sería, para ese número, la Se y PPV en hombres y mujeres?
"""
del pastX, pasty, pastXgroup, pastygroup
probs=globalmodel.predict_proba(X[features])[:,-1]
recallK,ppvK, specK, indices=performance(probs, np.where(y[config.COLUMNS]>=1,1,0), K)
selectedPatients=X.loc[indices]
selectedResponse=y.loc[indices]

print(f'Número de mujeres entre los {K} de mayor riesgo: ',sum(selectedPatients.FEMALE))

selectedfemale=selectedPatients['FEMALE']==1
selectedmale=selectedPatients['FEMALE']==0
probs[indices]
for i, group, sex, groupname in zip([1,0],[selectedfemale,selectedmale], [female, male],[ 'Mujeres','Hombres']):
    # SUBSET DATA
    print(groupname)
    Xgroup=X.loc[sex]
    ygroup=y.loc[sex]
    ytrue=np.where(ygroup[config.COLUMNS]>=1,1,0) #ytrue
    yy=y.loc[indices].loc[sex]  #selected women (ypred)
    yy.urgcms=1
    selectedSex=pd.DataFrame([0]*len(ygroup),index=ygroup.index)
    selectedSex['yy']=yy.urgcms
    ypred=selectedSex.yy.fillna(0)
    
    from sklearn.metrics import confusion_matrix
    c=confusion_matrix(y_true=ytrue, y_pred=ypred)
    print(c)
    TN, FP, FN, TP=c.ravel()
    print('TP, FP, FN, TN= ',TP, FP, FN, TN)
    recall=TP/(FN+TP)
    ppv=TP/(TP+FP)
    specificity = TN / (TN+FP)
    print('Recall, PPV, Spec = ',recall,ppv, specificity)