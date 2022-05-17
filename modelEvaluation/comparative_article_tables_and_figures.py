#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:24:55 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcms_excl_nbinj'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
from modelEvaluation.calibrate import calibrate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import roc_auc_score,average_precision_score, brier_score_loss,RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay

#%%
def model_labels(models):
    labels=[]
    #The abbreviations that should appear in tables and figures:
    dictionary={'logistic':'LR','randomForest':'RF',
                'neuralNetworkRandom':'MLP','hgb':'GBDT'}
    for m in models:
        alg=re.sub('_|[0-9]', '', m)
        labels.append(dictionary[alg])
    return labels

def ROC_PR_comparison(models, yr, logistic_model, mode='ROC', **kwargs):
    # load logistic predictions
    parent=calibrate(logistic_model, yr)
    display={}
    # load models predictions
    models.append(logistic_model)
    labels=model_labels(models)
    for m, label in zip(models, labels): 
        print(m, label)
        if m==logistic_model:
            model=calibrate(m, yr,experiment_name=config.EXPERIMENT,
                            )
        else:
            predpath=re.sub(config.EXPERIMENT,'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)
            model=calibrate(m, yr, experiment_name='hyperparameter_variability_'+config.EXPERIMENT,
                            filename=os.path.join(predpath,f'{m}_calibrated_{yr}.csv'))
        obs=np.where(model.OBS>=1,1,0)
        fpr, tpr, _ = roc_curve(obs, model.PREDCAL)
        prec, rec, _ = precision_recall_curve(obs, model.PREDCAL)
        roc_auc = auc(fpr, tpr)
        avg_prec = average_precision_score(obs, model.PREDCAL)
        if mode=='PR':
            display[label]=PrecisionRecallDisplay(prec, rec, 
                                     estimator_name=label,
                                     average_precision=avg_prec)
        elif mode=='ROC':
            display[label]= RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                      estimator_name=label)
    import matplotlib.pyplot as plt
    fig2, ax = plt.subplots(1,1,figsize=(10,10))
    for curve in display.values():
        curve.plot(ax)
    return(display)
def brier_boxplot(df, year, **kwargs):
    # path=kwargs.get('path',config.FIGUREPATH)
    # name=kwargs.get('name','')
    # X=kwargs.get('X',None)
    # y=kwargs.get('y',None)

    # df['Algorithm']=[re.sub('_|[0-9]', '', model) for model in df['Model'].values]
    parent_metrics=df.copy().loc[df.Algorithm=='logistic']
    df=df.loc[df.Algorithm!='logistic']
    fig=plt.figure()
    ax1 = plt.subplot2grid((3,2),(0,0))
    ax2 = plt.subplot2grid((3,2),(0,1))
    ax3 = plt.subplot2grid((3,2),(1,0))
    ax4 = plt.subplot2grid((3,2),(1,1))
    ax5 = plt.subplot2grid((3,2),(2,0), rowspan=1, colspan=2)
    # fig, ((ax1,ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(10,12), gridspec_kw={'width_ratios': [1,1,1,3]})
    plt.suptitle('')
    
    for metric, ax in zip(['Score', 'AP', 'Recall_20000', 'PPV_20000'],[ax1,ax2,ax3, ax4]):
        df.boxplot(column=metric, by='Algorithm', ax=ax)
        print(parent_metrics[metric].values[0])
        ax.axhline(y = parent_metrics[metric].values[0], linestyle = '-', label='Logistic', color='r')
    
    df['Before/After']='After'
    dff=df.copy()
    dff['Before/After']='Before'
    dff.Brier=dff['Brier Before']
    df2=pd.concat([dff,df])
    # df2['Before/After']='After'
    # df2.loc[original_index, 'Before/After']='Before'
    # df2.loc[original_index, 'Brier']=df2.loc[original_index, 'Brier Before']
    df2.boxplot(column='Brier', by=['Before/After','Algorithm'], ax=ax5)
    ax5.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r')
    plt.legend()
    # plt.savefig(os.path.join(path,f'hyperparameter_variability_{'Brier'}.png'))
    plt.show()
#%%

X,y=getData(2017)
X16,y17=getData(2016)
#%%
""" TABLE 1:  DEMOGRAPHICS"""
female=X['FEMALE']==1
male=X['FEMALE']==0
sex=[ 'Women','Men']
# table1={'Women':pd.DataFrame(), 'Men':pd.DataFrame()}
table1= pd.DataFrame(index=['N in 2017 (%)', 'Hospitalized in 2018', 
                            'Aged 0-17',
                            'Aged 18-64',
                            'Aged 65-69',
                            'Aged 70-79',
                            'Aged 80-84',
                            'Aged 85+'])
for group, groupname in zip([female,male],sex):
    print(groupname)
    Xgroup=X.loc[group]
    ygroup=y.loc[group]
    # ygroup18=y.loc[group18]
    a1=sum(Xgroup.AGE_0004)+sum(Xgroup.AGE_0511)+sum(Xgroup.AGE_0511)
    a2=sum(Xgroup.AGE_1834)+sum(Xgroup.AGE_3544)+sum(Xgroup.AGE_4554)+sum(Xgroup.AGE_5564)
    a3=sum(Xgroup.AGE_6569)
    a4=sum(Xgroup.AGE_7074)+sum(Xgroup.AGE_7579)
    a5=sum(Xgroup.AGE_8084)
    a85plus=len(Xgroup)-(a1+a2+a3+a4+a5)
    positives=sum(np.where(ygroup.urgcms>=1,1,0))
    table1[groupname]=[f'{len(Xgroup)} ({len(Xgroup)*100/len(X):2.2f} %)',
                       f'{positives} ({positives*100/len(Xgroup):2.2f} %)',
                       f'{a1} ({a1*100/len(Xgroup):2.2f} %) ',
                       f'{a2} ({a2*100/len(Xgroup):2.2f} %) ',
                       f'{a3} ({a3*100/len(Xgroup):2.2f} %) ',
                       f'{a4} ({a4*100/len(Xgroup):2.2f} %) ',
                       f'{a5} ({a5*100/len(Xgroup):2.2f} %) ',
                       f'{a85plus} ({a85plus*100/len(Xgroup):2.2f} %) ']

print(table1.to_latex())

simdif=len(set(X.PATIENT_ID.values).symmetric_difference(set(X16.PATIENT_ID.values)))

""" Comments on Table 1 """
print('The two populations contained...')
print(simdif, '(simetric difference)')
print(f'... patients not in common, that is, {simdif*100/len(X):2.2f} %')
print('Prevalence of admission in 2018:',sum(np.where(y.urgcms>=1,1,0))/len(X))
#%%
""" MATERIALS AND METHODS: Comments on variability assessment"""
K=20000
metrics=pd.read_csv(re.sub(config.EXPERIMENT, 'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)+'/metrics.csv')
print('Number of models per algorithm:')
print( metrics.groupby(['Algorithm'])['Algorithm'].count() )

""" RESULTS. TABLE 2 """
logisticMetrics=pd.read_csv(config.PREDPATH+'/metrics.csv')
logisticMetrics=logisticMetrics.loc[logisticMetrics.Model.str.startswith('logistic2022')]
logisticMetrics[f'F1_{K}']=2*logisticMetrics[f'Recall_{K}']*logisticMetrics[f'PPV_{K}']/(logisticMetrics[f'Recall_{K}']+logisticMetrics[f'PPV_{K}'])
logisticMetrics['Algorithm']=['logistic']

metrics=pd.concat([metrics, logisticMetrics])
#Discard some algorithms
metrics=metrics.loc[metrics.Algorithm.isin(('logistic','hgb','randomForest','neuralNetworkRandom'))]

# Option 1: Median --- (Interquartile range)
table2=pd.DataFrame()
from scipy.stats import iqr
for metric in ['Score', 'Recall_20000', 'PPV_20000', 'F1_20000','Brier']:
    median=metrics.groupby(['Algorithm'])[metric].median()
    IQR=metrics.groupby(['Algorithm'])[metric].agg(iqr)
    table2[metric]=[f'{m:1.3f} ({i:.2E})' for m, i in zip(median.values, IQR.values)]
    table2.index=IQR.index
  
print(table2.to_latex())

higher_better={'Score': True, 'Recall_20000': True,
               'PPV_20000': True, 'Brier':False, 'AP':True}
def use_f_3(x):
    return "%.4f" % x
def use_E(x):
    return "%.2e" % x
# Option 2: Subtables with descriptive
for metric in [ 'Score', 'Recall_20000', 'PPV_20000', 'Brier','AP']:
    table2=metrics.groupby(['Algorithm'])[metric].describe()[['25%','50%', '75%','std']].sort_values('50%', ascending=[not higher_better[metric]])
    print(table2.to_latex(formatters=[ use_f_3, use_f_3, use_f_3, use_E]))
    print('\n'*2)

def median(x):
    return x.quantile(0.5,interpolation='nearest')
median_models={}
for metric in ['Score', 'AP']:
    mediandf=metrics.groupby(['Algorithm'])[metric].agg([ median]).stack(level=0)
    for alg in metrics.Algorithm.unique():
            if alg=='logistic':
                continue
            df_alg=metrics.loc[metrics.Algorithm==alg].to_dict(orient='list')
            perc50=mediandf.loc[alg]['median']
            chosen_model=list(df_alg['Model'])[list(df_alg[metric]).index(perc50)]
            print(metrics.loc[metrics.Model==chosen_model][metric])
            try:
                median_models[metric].append(chosen_model)
            except KeyError:
                median_models[metric]=[chosen_model]
                


""" ROC AND PR FIGURES """
logistic_model='logistic20220207_122835'
ROC_PR_comparison(median_models['AP'], 2018, logistic_model, mode='PR')
ROC_PR_comparison(median_models['Score'], 2018, logistic_model, mode='ROC')

brier_boxplot(metrics, 2018)
