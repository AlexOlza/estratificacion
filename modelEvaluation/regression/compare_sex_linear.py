#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Compare model performance for men and women
Created on Tue Oct 4 2022

@author: aolza
"""
import os
import configurations.utility as util
from python_settings import settings as config
if not config.configured:
    experiment = input('Experiment: ')
    config_used = os.path.join(
        os.environ['USEDCONFIG_PATH'], f'{experiment}/linearMujeres.json')
    configuration = util.configure(config_used)
import joblib as job
from dataManipulation.dataPreparation import getData
from modelEvaluation.compare import detect_models, compare, detect_latest, performance
from modelEvaluation.predict import predict

import pandas as pd
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, r2_score, RocCurveDisplay, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
# %%


# %%


# def plot_roc(fpr, tpr, groupname):
#     roc_auc = auc(fpr, tpr)
#     display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                               estimator_name=groupname)
#     # display.plot()

#     return display


# def plot_pr(precision, recall, groupname, y, pred):
#     avg_prec = average_precision_score(y, pred)
#     display = PrecisionRecallDisplay(precision=precision, recall=recall,
#                                      estimator_name=groupname,
#                                      average_precision=avg_prec)
#     return display


def translateVariables(df, **kwargs):
    dictionaryFile = kwargs.get('file', os.path.join(
        config.INDISPENSABLEDATAPATH+'diccionarioACG.csv'))
    dictionary = pd.read_csv(dictionaryFile)
    dictionary.index = dictionary.codigo
    df = pd.merge(df, dictionary, right_index=True, left_index=True)
    return df


# %%
year = int(input('YEAR TO PREDICT: '))
X, y = getData(year-1)
try:
    pastX, pasty = getData(year-2)
except KeyError:
    pastX, pasty =X,y
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
sex = ['Mujeres',  'Hombres']
# %%
# tail=True
# if tail:
#     xmin, xmax= 0.5, 1
# else:
#     xmin, xmax= -0.02, 0.23
separate_cost, joint_cost = {}, {}

table = pd.DataFrame()
K = 20000
models = ['Global', 'Separado']
fighist, axs = plt.subplots(2, 2, figsize=(25, 20))

axhist, axhist2, axhist3, axhist4= axs[0,0], axs[0,1], axs[1,0], axs[1,1]
# fighist2, (axhist3, axhist4) = plt.subplots(1, 2)

for i, group, groupname in zip([1, 0], [female, male], sex):
    recall, ppv, spec, score, ap = {}, {}, {}, {}, {}
    selected = [l for l in available_models if (bool(re.match(f'linear{groupname}|linear\d+', l)))]
    print('Selected models: ', selected)
    # LOAD MODELS
    globalmodelname = list(
        set(selected)-set([f'linear{groupname}']))[0]
    separatemodelname = f'linear{groupname}'
    globalmodel = job.load(config.MODELPATH+globalmodelname+'.joblib')
    separatemodel = job.load(config.MODELPATH+separatemodelname+'.joblib')

    # SUBSET DATA
    Xgroup = X.loc[group]
    ygroup = y.loc[y.PATIENT_ID.isin(Xgroup.PATIENT_ID)]

    pastXgroup = pastX.loc[pastX['FEMALE'] == i]
    pastygroup = pasty.loc[pasty.PATIENT_ID.isin(pastXgroup.PATIENT_ID)]

    assert (all(Xgroup['FEMALE'] == 1) or all(Xgroup['FEMALE'] == 0))

    # PREDICT
    joint_preds, joint_r2= predict(globalmodelname,config.EXPERIMENT,year,
                                   X=Xgroup,y=ygroup, filename=f'{globalmodelname}{groupname}')
    from time import time
    t0=time()
    separate_preds, separate_r2= predict(separatemodelname,config.EXPERIMENT,year,
                                         X=Xgroup.drop('FEMALE',axis=1),y=ygroup)

    print(f'pred time {groupname} ',time()-t0)
    
    import seaborn as sns
    # plt.xlim(0, 0.2)
    # axhist.set_xlim(xmin,xmax)
    # axhist2.set_xlim(xmin,xmax)
    # if tail:
    #     axhist.set_ylim(0,0.1)
    #     axhist2.set_ylim(0,0.1)
    sns.kdeplot(separate_preds.PRED, shade=True, ax=axhist,
                 label=groupname, bw_method=0.3)
    sns.kdeplot(joint_preds.PRED, shade=True, ax=axhist2,
                 label=groupname, bw_method=0.3)

    axhist.set_title('Modelos separados')
    axhist2.set_title('Modelo global')
    # plt.legend()
    # plt.tight_layout()

    ax = axhist3 if groupname == 'Mujeres' else axhist4
    # ax.set_xlim(xmin,xmax)
    # if tail:
    #     ax.set_ylim(0,0.1)
    sns.kdeplot(separate_preds.PRED, shade=True, ax=ax,
                 label='Separados', bw_method=0.3)
    sns.kdeplot(joint_preds.PRED, shade=True, ax=ax,
                 label='Global', bw_method=0.3)
    ax.set_title(groupname)
    
    obs=separate_preds.OBS

    # METRICS
    for model, preds in zip(models, [joint_preds, separate_preds]):
        rsquared=r2_score(y_true=preds['OBS'].values, y_pred=preds['PRED'].values)
        
        if rsquared<0:
            problematic=preds.iloc[((preds['PRED']-preds['OBS'])**2).nlargest(6).index].PATIENT_ID
            unproblematic=separate_preds[~separate_preds.PATIENT_ID.isin(problematic)]
            rsquared=r2_score(unproblematic.OBS, unproblematic.PRED)

        
        recallK, ppvK, specK, _ = performance(obs, preds['PRED'], K)
        recall[model] = recallK
        spec[model] = specK
        score[model] = rsquared
        ppv[model] = ppvK

    df = pd.DataFrame(list(zip([f'{model}- {groupname}' for model in models],
                               [score[model] for model in models],
                               
                               [recall[model] for model in models],
                               [spec[model] for model in models],
                               [ppv[model] for model in models])),
                      columns=['Model', 'R2',  f'Recall_{K}', f'Specificity_{K}', f'PPV_{K}'])
    table = pd.concat([df, table])

axhist2.legend()
axhist4.legend()
axhist3.legend()
axhist.legend()
plt.tight_layout()

# %%
print(table.to_markdown(index=False))

#%%

