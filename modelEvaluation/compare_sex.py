#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Compare model performance for men and women
Created on Fri Mar 18 13:13:37 2022

@author: alex
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

import joblib as job
from dataManipulation.dataPreparation import getData
import modelEvaluation.calibrate as cal
from modelEvaluation.compare import detect_models, compare, detect_latest, performance


import pandas as pd
import re
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,average_precision_score, roc_auc_score, RocCurveDisplay, roc_curve,  precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import auc as auc_function
import sys
from sklearn.metrics import mean_squared_error,r2_score
sys.path.append('/home/alex/Desktop/estratificacion/')
# %%


# %%


def plot_roc(fpr, tpr, groupname):
    roc_auc = auc_function(fpr, tpr)
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


def translateVariables(df, **kwargs):
    dictionaryFile = kwargs.get('file', os.path.join(
        config.INDISPENSABLEDATAPATH+'diccionarioACG.csv'))
    dictionary = pd.read_csv(dictionaryFile)
    dictionary.index = dictionary.codigo
    df = pd.merge(df, dictionary, right_index=True, left_index=True)
    return df
from sklearn.metrics import recall_score
def precision_at_recall(precision, recall, threshold, modelname, value):
    df = pd.DataFrame()
    df['precision'] = precision[:-1]
    df['recall'] = recall[:-1]
    # df['specificity'] = specificity[:-1]
    df['threshold'] = threshold
    df['model'] = modelname    
    return(df.iloc[(df['recall']-value).abs().argsort()[:1]])

def T1a_T2a(preds,K=20000):
    T=[]
    if config.ALGORITHM=='logistic':
        preds['NEWOBS']=np.where(preds.OBS>=1,1,0)
    else:
        preds['NEWOBS']=np.where(preds.OBS.isin(preds.OBS.nlargest(K)), 1, 0)
    for val, df in preds.groupby('FEMALE'):
        obs=df.NEWOBS
        if config.ALGORITHM=='logistic':
            score1=roc_auc_score(obs, df.PRED) #auc
            score2=average_precision_score(obs, df.PRED) #ap
            cols=['FEMALE','AUC','AP','Recall_K','PPV_K']
        else:
            score1=r2_score(df.OBS, df.PRED)
            score2=mean_squared_error(df.OBS, df.PRED,squared=False)
            
            cols=['FEMALE','R2','RMSE','Recall_K','PPV_K']
        c = confusion_matrix(y_true=obs, y_pred=df.top_K)
        recall = c[1][1] / (c[1][0] + c[1][1])
        ppv = c[1][1] / (c[0][1] + c[1][1])
        T.append([val,score1,score2, recall, ppv])
    T=pd.DataFrame(T,columns=cols)
    T['Percentage']=[100*len(preds.loc[preds.top_K==1].loc[preds.loc[preds.top_K==1]['FEMALE']==val])/K for val in T.FEMALE.values]
    return T
def tables_gender_article_draft(K=20000):
    if config.ALGORITHM=='logistic':
        models={'global':'logistic20230324_111354',
                'separated':['logisticMujeres','logisticHombres'],
                'same_prevalence':'logistic_gender_balanced'}
        #T1a
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname]=cal.calibrate(models["global"], year,
                                             filename=f'{models["global"]}_{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds['top_K']=np.where(
            preds.PATIENT_ID.isin(preds.nlargest(K,'PRED').PATIENT_ID),1,0)
        T1a=T1a_T2a(preds)
        
        #T2a
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname]=cal.calibrate(f'logistic{groupname}', year,
                                             filename=f'logistic{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds['top_K']=np.where(
            preds.PATIENT_ID.isin(preds.nlargest(K,'PRED').PATIENT_ID),1,0)
        
        T2a=T1a_T2a(preds)
        
        #T3a tomar 10k y 10k
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname]=cal.calibrate(f'logistic{groupname}', year,
                                             filename=f'logistic{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
            preds[groupname]['top_K']=np.where(
                preds[groupname].PATIENT_ID.isin(preds[groupname].nlargest(int(K/2),'PRED').PATIENT_ID),1,0)
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
       
        T3a=T1a_T2a(preds)
        
        #T4
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname]=cal.calibrate(f'logistic_gender_balanced', year,
                                             filename=f'logistic_gender_balanced_{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds['top_K']=np.where(
            preds.PATIENT_ID.isin(preds.nlargest(K,'PRED').PATIENT_ID),1,0)
        
        T4=T1a_T2a(preds)
        
    else:
        models={'global':'linear20230324_130625',
                'separated':['linearMujeres','linearHombres']}
        
        #T1b
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname],_=predict(models["global"],config.EXPERIMENT, year,
                                             filename=f'{models["global"]}_{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds['top_K']=np.where(
            preds.PATIENT_ID.isin(preds.nlargest(K,'PRED').PATIENT_ID),1,0)
        preds.loc[preds.PRED<0,'PRED']=0
        T1a=T1a_T2a(preds)
        
        #T2a
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname],_=predict(f'linear{groupname}',config.EXPERIMENT, year,
                                             filename=f'linear{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds['top_K']=np.where(
            preds.PATIENT_ID.isin(preds.nlargest(K,'PRED').PATIENT_ID),1,0)
        preds.loc[preds.PRED<0,'PRED']=0
        problematic=((preds.PRED-preds.OBS)**2).nlargest(10).index
        unproblematic_preds=preds.loc[~ preds.index.isin(problematic)]
     
        T2a=T1a_T2a(unproblematic_preds)
        
        #T3a tomar 10k y 10k
        preds={}
        for groupname in ['Mujeres', 'Hombres']:
            preds[groupname],_=predict(f'linear{groupname}',config.EXPERIMENT, year,
                                             filename=f'linear{groupname}')
            preds[groupname]['FEMALE']=1 if groupname=='Mujeres' else 0
            preds[groupname]['top_K']=np.where(
                preds[groupname].PATIENT_ID.isin(preds[groupname].nlargest(int(K/2),'PRED').PATIENT_ID),1,0)
        preds=pd.concat([preds['Mujeres'], preds['Hombres']])
        preds.loc[preds.PRED<0,'PRED']=0
        print(((preds.PRED-preds.OBS)**2).nlargest(10))
        problematic=((preds.PRED-preds.OBS)**2).nlargest(10).index
        unproblematic_preds=preds.loc[~ preds.index.isin(problematic)]
        
        T3a=T1a_T2a(unproblematic_preds)
        T4=pd.DataFrame()
    return {'T1a':T1a,'T2a':T2a,'T3a':T3a,'T4':T4}
        
# %%
year = int(input('YEAR TO PREDICT: '))
X, y = getData(year-1)
pastX, pasty = getData(year-2)
# %%
# X.drop('PATIENT_ID', axis=1, inplace=True)
predictors = X.columns
features = X.drop('PATIENT_ID', axis=1).columns

# for column in features:
#     if column != 'FEMALE':
#         X[f'{column}INTsex'] = X[column]*X['FEMALE']
#         pastX[f'{column}INTsex'] = pastX[column]*pastX['FEMALE']
# %%
available_models = detect_models()
# latest=detect_latest(available_models)

female = X['FEMALE'] == 1
male = X['FEMALE'] == 0
sex = ['Mujeres', 'Hombres']

#%%
separate,joint={},{}
if config.ALGORITHM=='linear':
    from modelEvaluation.predict import predict
    for i, group, groupname in zip([1, 0], [female, male], sex):
        recall, ppv, spec, r2, rmse = {}, {}, {}, {}, {}
        selected = [l for l in available_models if (bool(re.match(f'{config.ALGORITHM}Random$|{config.ALGORITHM}{groupname}$|{config.ALGORITHM}\d+', l)))]
        print('Selected models: ', selected)
        # LOAD MODELS
        globalmodelname = list(
            set(selected)-set([f'{config.ALGORITHM}{groupname}']))[0]
        separatemodelname = f'{config.ALGORITHM}{groupname}'

        globalmodel = job.load(config.MODELPATH+globalmodelname+'.joblib')
        separatemodel = job.load(config.MODELPATH+separatemodelname+'.joblib')
        # SUBSET DATA
        Xgroup = X.loc[group]
        ygroup = y.loc[y.PATIENT_ID.isin(Xgroup.PATIENT_ID)]

        pastXgroup = pastX.loc[pastX['FEMALE'] == i]
        pastygroup = pasty.loc[pasty.PATIENT_ID.isin(pastXgroup.PATIENT_ID)]

        assert (all(Xgroup['FEMALE'] == 1) or all(Xgroup['FEMALE'] == 0))


        # PREDICT
        separate[groupname],_ = predict(f'{config.ALGORITHM}{groupname}',config.EXPERIMENT, year,
                                                filename=f'{config.ALGORITHM}{groupname}',
                                                predictors=[
                                                    p for p in predictors if not p == 'FEMALE'],
                                                X=Xgroup[predictors].drop(
                                                    'FEMALE', axis=1),
                                                y=ygroup,
                                                )

        joint[groupname],_ = predict(globalmodelname,config.EXPERIMENT, year,
                                             filename=f'{globalmodelname}_{groupname}',
                                             predictors=predictors,
                                             X=Xgroup[predictors], y=ygroup,
                                             )
        
    tables=tables_gender_article_draft()
    
    separate_preds=separate['Hombres']
    for table in tables.values():
        print(table.round(2).to_latex(index=False))
        print('\n')
    quit()
# %%
tail=True
if tail:
    xmin, xmax= 0.5, 1
else:
    xmin, xmax= -0.02, 0.23
separate_cal, joint_cal, balanced_cal = {}, {}, {}
roc, roc_joint, roc_sameprev = {k: {} for k in sex}, {
    k: {} for k in sex}, {k: {} for k in sex}
pr, pr_joint, pr_sameprev = {k: {} for k in sex}, {
    k: {} for k in sex}, {k: {} for k in sex}  # precision-recall curves
PRECISION, RECALL, THRESHOLDS = {k: {} for k in sex}, {
    k: {} for k in sex}, {k: {} for k in sex}
FPR, TPR, ROCTHRESHOLDS = {k: {} for k in sex}, {k: {}
                                                 for k in sex}, {k: {} for k in sex}
prec_at_rec=pd.DataFrame()
table = pd.DataFrame()
K = 20000
models = ['Global', 'Separado', 'Same prevalence']
fighist, axs = plt.subplots(2, 2, figsize=(25, 20))

axhist, axhist2, axhist3, axhist4= axs[0,0], axs[0,1], axs[1,0], axs[1,1]
# fighist2, (axhist3, axhist4) = plt.subplots(1, 2)

for i, group, groupname in zip([1, 0], [female, male], sex):
    recall, ppv, spec, score, ap = {}, {}, {}, {}, {}
    selected = [l for l in available_models if (bool(re.match(f'{config.ALGORITHM}Random$|{config.ALGORITHM}{groupname}$|{config.ALGORITHM}\d+|logistic_gender_balanced', l)))]
    print('Selected models: ', selected)
    # LOAD MODELS
    globalmodelname = list(
        set(selected)-set([f'{config.ALGORITHM}{groupname}'])-set([f'logistic_gender_balanced']))[0]
    separatemodelname = f'{config.ALGORITHM}{groupname}'
    if not 'neural' in config.ALGORITHM:
        globalmodel = job.load(config.MODELPATH+globalmodelname+'.joblib')
        try:
            sameprevmodel = job.load(
                config.MODELPATH+f'{config.ALGORITHM}_gender_balanced.joblib')
            sameprev=True
        except FileNotFoundError:
            print('Same prevalence model not found')
            sameprev=None
        separatemodel = job.load(config.MODELPATH+separatemodelname+'.joblib')
    else:
        globalmodel = keras.models.load_model(config.MODELPATH+globalmodelname)
        separatemodel = keras.models.load_model(config.MODELPATH+separatemodelname) 
        sameprev=None
    # SUBSET DATA
    Xgroup = X.loc[group]
    ygroup = y.loc[y.PATIENT_ID.isin(Xgroup.PATIENT_ID)]

    pastXgroup = pastX.loc[pastX['FEMALE'] == i]
    pastygroup = pasty.loc[pasty.PATIENT_ID.isin(pastXgroup.PATIENT_ID)]

    assert (all(Xgroup['FEMALE'] == 1) or all(Xgroup['FEMALE'] == 0))

    pos = sum(np.where(ygroup[config.COLUMNS] >= 1, 1, 0))
    pastpos = sum(np.where(pastygroup[config.COLUMNS] >= 1, 1, 0))
    print('Sample size 2017', len(Xgroup), 'positive: ',
          pastpos, 'prevalence=', pastpos/len(pastXgroup))
    print('Sample size 2018', len(pastXgroup),
          'positive: ', pos, 'prevalence=', pos/len(Xgroup))

    # PREDICT
    separate_cal[groupname] = cal.calibrate(f'{config.ALGORITHM}{groupname}', year,
                                            filename=f'{config.ALGORITHM}{groupname}',
                                            predictors=[
                                                p for p in predictors if not p == 'FEMALE'],
                                            presentX=Xgroup[predictors].drop(
                                                'FEMALE', axis=1),
                                            presentY=ygroup,
                                            pastX=pastXgroup[predictors].drop(
                                                'FEMALE', axis=1),
                                            pastY=pastygroup)

    joint_cal[groupname] = cal.calibrate(globalmodelname, year,
                                         filename=f'{globalmodelname}_{groupname}',
                                         predictors=predictors,
                                         presentX=Xgroup[predictors], presentY=ygroup,
                                         pastX=pastXgroup[predictors], pastY=pastygroup)
    # balanced_cal[groupname] = joint_cal[groupname] #provisional
    if sameprev:
        balanced_cal[groupname] = cal.calibrate(f'{config.ALGORITHM}_gender_balanced', year,
                                                filename=f'{config.ALGORITHM}_gender_balanced_{groupname}',
                                                presentX=Xgroup[predictors], presentY=ygroup,
                                                pastX=pastXgroup[predictors],
                                                pastY=pastygroup,
                                                )
        
    else:
        print('Omitting same prevalence model, we copy the global model results')
        balanced_cal[groupname] =cal.calibrate(globalmodelname, year,
                                             filename=f'{globalmodelname}_{groupname}',
                                             predictors=predictors,
                                             presentX=Xgroup[predictors], presentY=ygroup,
                                             pastX=pastXgroup[predictors], pastY=pastygroup)

    balanced_preds = balanced_cal[groupname].PRED
    joint_preds = joint_cal[groupname].PRED
    separate_preds = separate_cal[groupname].PRED

    obs = np.where(balanced_cal[groupname].OBS >= 1, 1, 0)

    assert all(balanced_cal[groupname].OBS == joint_cal[groupname].OBS)
    
    import seaborn as sns
    # plt.xlim(0, 0.2)
    axhist.set_xlim(xmin,xmax)
    axhist2.set_xlim(xmin,xmax)
    if tail:
        axhist.set_ylim(0,0.1)
        axhist2.set_ylim(0,0.1)
    sns.kdeplot(separate_preds, shade=True, ax=axhist,
                clip=(0, 1), label=groupname, bw_method=0.3)
    sns.kdeplot(joint_preds, shade=True, ax=axhist2,
                clip=(0, 1), label=groupname, bw_method=0.3)

    axhist.set_title('Modelos separados')
    axhist2.set_title('Modelo global')
    # plt.legend()
    # plt.tight_layout()

    ax = axhist3 if groupname == 'Mujeres' else axhist4
    ax.set_xlim(xmin,xmax)
    if tail:
        ax.set_ylim(0,0.1)
    sns.kdeplot(separate_preds, shade=True, ax=ax,
                clip=(0, 1), label='Separados', bw_method=0.3)
    sns.kdeplot(joint_preds, shade=True, ax=ax,
                clip=(0, 1), label='Global', bw_method=0.3)
    ax.set_title(groupname)
    

    # METRICS
    for model, preds in zip(models, [joint_preds, separate_preds, balanced_preds]):
        prec, rec, thre = precision_recall_curve(obs, preds)
        # spec = [recall_score(obs, preds >= t, pos_label=0) for t in thre]
        fpr, tpr, rocthresholds = roc_curve(obs, preds)
        FPR[groupname][model] = fpr
        TPR[groupname][model] = tpr
        PRECISION[groupname][model] = prec
        RECALL[groupname][model] = rec
        THRESHOLDS[groupname][model] = thre
        ROCTHRESHOLDS[groupname][model] = rocthresholds
        # CURVES - PLOTS
        roc[groupname][model] = plot_roc(fpr, tpr, groupname)
        pr[groupname][model] = plot_pr(prec, rec, groupname, obs, preds)

        recallK, ppvK, specK, _ = performance(obs, preds, K)
        rocauc = roc_auc_score(obs, preds)
        avg_prec = average_precision_score(obs, preds)
        recall[model] = recallK
        spec[model] = specK
        score[model] = rocauc
        ppv[model] = ppvK
        ap[model] = avg_prec
        
        prec_at_rec_model=precision_at_recall(prec,rec,thre,f'{model}- {groupname}',0.14)
        t=prec_at_rec_model.threshold.values[0]
        specif = [recall_score(obs, preds >= t, pos_label=0)]
        prec_at_rec_model['specificity']=specif

        prec_at_rec=pd.concat([prec_at_rec,prec_at_rec_model])
        print(prec_at_rec)
        
    df = pd.DataFrame(list(zip([f'{model}- {groupname}' for model in models],
                               [score[model] for model in models],
                               [ap[model] for model in models],
                               [recall[model] for model in models],
                               [spec[model] for model in models],
                               [ppv[model] for model in models])),
                      columns=['Model', 'AUC', 'AP', f'Recall_{K}', f'Specificity_{K}', f'PPV_{K}'])
    table = pd.concat([df, table])

axhist2.legend()
axhist4.legend()
axhist3.legend()
axhist.legend()
plt.tight_layout()
# %%
fig1, (ax_sep1, ax_joint1, ax_sameprev1) = plt.subplots(1, 3, figsize=(16, 8))
fig2, (ax_sep2, ax_joint2, ax_sameprev2) = plt.subplots(1, 3, figsize=(16, 8))
for groupname in sex:
    for ax, model in zip((ax_sep1, ax_joint1, ax_sameprev1), models):
        roc[groupname][model].plot(ax)
        ax.set_title(model)
    for ax, model in zip((ax_sep2, ax_joint2, ax_sameprev2), models):
        pr[groupname][model].plot(ax)
        ax.set_title(model)
fig1.savefig(os.path.join(config.FIGUREPATH, 'rocCurve.png'))
fig2.savefig(os.path.join(config.FIGUREPATH, 'prCurve.png'))
plt.show()
print(table.to_markdown(index=False,))
print(prec_at_rec.to_markdown(index=False,))
# %%
# %%
for i, model in enumerate(models):
    print(' ')
    print(model)
    print('what should the probability threshold be to get the same recall for women as for men? ')
    print('Recall for men (global model): ', table.Recall_20000.iloc[i])
    idx = [n for n, i in enumerate(
        RECALL['Mujeres'][model]) if i <= table.Recall_20000.iloc[0]][0]
    t = THRESHOLDS['Mujeres'][model][idx]
    idx2 = [n for n, i in enumerate(
        ROCTHRESHOLDS['Mujeres'][model]) if i <= t][0]
    print('Threshold: ', t)
    print('Recall: ', RECALL['Mujeres'][model][idx])
    print('Number of selected women: ', sum(joint_preds >= t))

    print('Specificity for these women is: ', 1-FPR['Mujeres'][model][idx2])

    print('And for men: ', table.Specificity_20000.iloc[i])
    print('PPV for these women is: ', PRECISION['Mujeres'][model][idx])
    print('ANd for men: ', table.PPV_20000.iloc[i])

# %% CALIBRATION CURVES
for title, preds in zip(['Global', 'Separado', 'Misma Prevalencia'], [joint_cal, separate_cal, balanced_cal]):
    cal.plot(preds, filename=title, consistency_bars=False)
# %% CALIBRATION CURVES (ALL MODELS TOGETHER)

all_preds = {'Global- Hombres': joint_cal['Hombres'],
             'Global- Mujeres': joint_cal['Mujeres'],
             'Separado- Mujeres': separate_cal['Mujeres'],
             'Separado- Hombres': separate_cal['Hombres']}
fname = 'GlobalSeparado_'
cal.plot(all_preds, filename=fname, consistency_bars=False)
# %%
"""
Si empleamos el modelo global y seleccionamos a los 20000 de mayor riesgo
(independientemente de si son hombres o mujeres) 
¿Cual sería el número de hombres y mujeres seleccionados? 
¿Cual sería, para ese número, la Se y PPV en hombres y mujeres?
"""
# del pastX, pasty, pastXgroup, pastygroup
probs = globalmodel.predict_proba(X[features])[:, -1]
recallK, ppvK, specK, indices = performance(
    pred=probs, obs=np.where(y[config.COLUMNS] >= 1, 1, 0), K=K)
selectedPatients = X.loc[indices.ravel()]
selectedResponse = y.loc[indices.ravel()]

print(f'Número de mujeres entre los {K} de mayor riesgo: ', sum(
    selectedPatients.FEMALE))

selectedfemale = selectedPatients['FEMALE'] == 1
selectedmale = selectedPatients['FEMALE'] == 0
probs[indices]
for i, group, sex, groupname in zip([1, 0], [selectedfemale, selectedmale], [female, male], ['Mujeres', 'Hombres']):
    # SUBSET DATA
    print(groupname)
    Xgroup = X.loc[sex]
    ygroup = y.loc[sex]
    ytrue = np.where(ygroup[config.COLUMNS] >= 1, 1, 0)  # ytrue
    yy = y.loc[indices].loc[sex]  # selected women (ypred)
    yy.urgcms = 1
    selectedSex = pd.DataFrame([0]*len(ygroup), index=ygroup.index)
    selectedSex['yy'] = yy.urgcms
    ypred = selectedSex.yy.fillna(0)

    from sklearn.metrics import confusion_matrix
    c = confusion_matrix(y_true=ytrue, y_pred=ypred)
    print(c)
    TN, FP, FN, TP = c.ravel()
    print('TP, FP, FN, TN= ', TP, FP, FN, TN)
    recall = TP/(FN+TP)
    ppv = TP/(TP+FP)
    specificity = TN / (TN+FP)
    print('Recall, PPV, Spec = ', recall, ppv, specificity)

#%%
tables=tables_gender_article_draft()

for table in tables.values():
    print(table.round(2).to_markdown(index=False))
    print('\n')