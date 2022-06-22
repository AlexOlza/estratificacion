#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:25:11 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
logistic_model='logistic20220207_122835'
chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcms_excl_nbinj'

import importlib
importlib.invalidate_caches()
from python_settings import settings as config

if not config.configured:
    logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import modelEvaluation.calibrate as cal
import pandas as pd
import numpy as np
import re
from modelEvaluation.comparative_article_plotting_functions import *
#%%

#%%
yr=2018
X,y=getData(2017)
# X16,y17=getData(2016)
#%%

#%%
""" MATERIALS AND METHODS: Comments on variability assessment"""
K=20000

test_metrics=pd.read_csv(re.sub(config.EXPERIMENT, 'hyperparameter_variability_urgcms_excl_nbinj',config.PREDPATH)+'/metrics2018.csv')
test_metrics=test_metrics.loc[test_metrics.Algorithm.isin(('logistic','hgb','randomForest','neuralNetworkRandom'))]
test_logisticMetrics=pd.read_csv(config.PREDPATH+'/metrics2018.csv')
test_logisticMetrics=test_logisticMetrics.loc[test_logisticMetrics.Algorithm=='logistic']
# test_logisticMetrics.Algorithm=['logistic 2018']
test_metrics=pd.concat([test_metrics,test_logisticMetrics])
test_metrics['Year']=2018
# test_logisticMetrics.Algorithm=test_logisticMetrics.Algorithm+' Test'

metrics=pd.read_csv(re.sub(config.EXPERIMENT, 'hyperparameter_variability_urgcms_excl_nbinj',config.PREDPATH)+'/metrics2017.csv')
print('Number of models per algorithm:')
print( metrics.groupby(['Algorithm'])['Algorithm'].count() )

""" RESULTS. TABLE 2 """
logisticMetrics=pd.read_csv(config.PREDPATH+'/metrics2017.csv')
logisticMetrics=logisticMetrics.loc[logisticMetrics.Model.str.startswith('logistic2022')]
logisticMetrics[f'F1_{K}']=2*logisticMetrics[f'Recall_{K}']*logisticMetrics[f'PPV_{K}']/(logisticMetrics[f'Recall_{K}']+logisticMetrics[f'PPV_{K}'])
logisticMetrics['Algorithm']=['logistic']

metrics=pd.concat([metrics, logisticMetrics])
#Discard some algorithms
metrics=metrics.loc[metrics.Algorithm.isin(('logistic','hgb','randomForest','neuralNetworkRandom'))]
metrics['Year']=2017

allmetrics=pd.concat([metrics, test_metrics])
# Option 1: Median --- (Interquartile range)
table2=pd.DataFrame()
from scipy.stats import iqr
for metric in ['Score', 'Recall_20000', 'PPV_20000', 'F1_20000','Brier']:
    median=metrics.groupby(['Algorithm'])[metric].median()
    IQR=metrics.groupby(['Algorithm'])[metric].agg(iqr)
    table2[metric]=[f'{m:1.3f} ({i:.2E})' for m, i in zip(median.values, IQR.values)]
    table2.index=IQR.index
  
print(table2.style.to_latex())

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

#%%
""" RESULTS - COMMENTS """
from modelEvaluation.compare import performance
median_models={}
correct={}
incorrect={}
for metric in ['Recall_20000', 'PPV_20000']:
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
    correct[metric]={}     
    incorrect[metric]={}
    for model in median_models[metric]+[logistic_model]:
        if model==logistic_model:
            preds= cal.calibrate(model, yr)
        else:
            predpath=re.sub(config.EXPERIMENT,'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)
            preds= cal.calibrate(model, yr,  experiment_name='hyperparameter_variability_urgcms_excl_nbinj',
                                                           filename=os.path.join(predpath,f'{model}_calibrated_{yr}.csv'))
        tn, fp, fn, tp=performance(preds.OBS,preds.PREDCAL,K,computemetrics=False)
        correct[metric][model]=tn+tp
        incorrect[metric][model]=fn+fp
#%%
for metric in ['Recall_20000', 'PPV_20000']:
    print('MLP vs RF ' , correct[metric]['neuralNetworkRandom_67']-correct[metric]['randomForest_29'])       
    print('MLP vs LR ' , correct[metric]['neuralNetworkRandom_67']-correct[metric][logistic_model])       

#%%

median_models = {}
for metric in ['Score', 'AP']:
    mediandf = metrics.groupby(['Algorithm'])[metric].agg([median]).stack(level=0)
    for alg in metrics.Algorithm.unique():
        if alg == 'logistic':
            continue
        df_alg = metrics.loc[metrics.Algorithm == alg].to_dict(orient='list')
        perc50 = mediandf.loc[alg]['median']
        chosen_model = list(df_alg['Model'])[list(df_alg[metric]).index(perc50)]
        print(metrics.loc[metrics.Model == chosen_model][metric])
        try:
            median_models[metric].append(chosen_model)
        except KeyError:
            median_models[metric] = [chosen_model]

""" ROC AND PR FIGURES """
# os.environ["DISPLAY"] =':99'
ROC_PR_comparison(median_models['AP'], 2018, logistic_model, mode='PR')
ROC_PR_comparison(median_models['Score'], 2018, logistic_model, mode='ROC')

""" BOXPLOTS """

# for violin in (True, False):
#     for together in (True, False):
boxplots(allmetrics, violin=True, together=False, hue='Year',supplementary=True)

#%% 
""" CHARACTERISTICS OF THE RISK GROUP """
metrics=pd.read_csv(re.sub(config.EXPERIMENT, 'hyperparameter_variability_urgcms_excl_nbinj',config.PREDPATH)+'/metrics2018.csv')
logisticMetrics=pd.read_csv(config.PREDPATH+'/metrics2018.csv')
logisticMetrics=logisticMetrics.loc[logisticMetrics.Model.str.startswith('logistic2022')]
logisticMetrics['Algorithm']=['logistic']

metrics=pd.concat([metrics, logisticMetrics])
#Discard some algorithms
metrics=metrics.loc[metrics.Algorithm.isin(('logistic','hgb','randomForest','neuralNetworkRandom'))]
preds={}
median_models={}
risk_groups={}
for metric in ['PPV_20000']:
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
    for model in median_models[metric]+[logistic_model]:
        print(model)
        if model==logistic_model:
            preds[model]= cal.calibrate(model, yr)
        else:
            predpath=re.sub(config.EXPERIMENT,'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)
            preds[model]= cal.calibrate(model, 2018,  experiment_name='hyperparameter_variability_urgcms_excl_nbinj',
                                                           filename=os.path.join(predpath,f'{model}_calibrated_2018.csv'))
        risk_group=preds[model].nlargest(20000, 'PREDCAL')
        print(risk_group.head())
        risk_groups[model]=risk_group.PATIENT_ID.values
        

risk_groups['General Population']=X.PATIENT_ID.values
#%%
table=pd.DataFrame()
columns=['neuralNetworkRandom_43', logistic_model,'General Population']
for chosen_model in columns:
    Xx=X.loc[X.PATIENT_ID.isin(risk_groups[chosen_model])]
    Yy=y.loc[y.PATIENT_ID.isin(risk_groups[chosen_model])]
    
    comorbidities={'COPD': ['EDC_RES04'],
                    'Chronic Renal Failure': ['EDC_REN01', 'EDC_REN06'],
                    'Heart Failure': ['EDC_CAR05'],
                    'Depression': ['EDC_PSY09', 'EDC_PSY20'],
                    'Diabetes Mellitus': ['EDC_END06','EDC_END07','EDC_END08', 'EDC_END09'],
                    # 'Dis. of Lipid Metabolism': ['EDC_CAR11'],
                    'Hypertension': ['EDC_CAR14','EDC_CAR15'],
                    'Ischemic Heart Disease': ['EDC_CAR03'],
                    'Low back pain': ['EDC_MUS14'],
                    'Osteoporosis': ['EDC_END02'],
                    "Parkinson's disease":['EDC_NUR06'],
                    'Persistent asthma':['EDC_ALL05', 'EDC_ALL04'],
                    'Rheumatoid arthritis':['EDC_RHU05'],
                    'Schizophrenia & affective dis.': ['EDC_PSY07'],
                    'Seizure disorders': ['EDC_NUR07']
                    }
    
    table1= pd.DataFrame(index=[ 'Hospitalized in 2018', 
                                '% of women',
                                'Aged 0-17',
                                'Aged 18-64',
                                'Aged 65-69',
                                'Aged 70-79',
                                'Aged 80-84',
                                'Aged 85+']+list(comorbidities.keys()))
    comorb={}
    
    
    comorb=[]
    for disease, EDClist in comorbidities.items():
        s=[Xx[EDC].sum() for EDC in EDClist]
        comorb.append(f'{sum(s)} ({sum(s)*100/len(Xx):2.2f} %)')
        print(disease, 'total (M+W): ',sum([Xx[EDC].sum() for EDC in EDClist]))
    # Yy18=y.loc[group18]
    a1=sum(Xx.AGE_0004)+sum(Xx.AGE_0511)+sum(Xx.AGE_0511)
    a2=sum(Xx.AGE_1834)+sum(Xx.AGE_3544)+sum(Xx.AGE_4554)+sum(Xx.AGE_5564)
    a3=sum(Xx.AGE_6569)
    a4=sum(Xx.AGE_7074)+sum(Xx.AGE_7579)
    a5=sum(Xx.AGE_8084)
    a85plus=len(Xx)-(a1+a2+a3+a4+a5)
    truepositives=sum(np.where(Yy.urgcms>=1,1,0))
    allpositives=sum(np.where(y.urgcms>=1,1,0))
    falsepositives=sum(np.where(Yy.urgcms>=1,0,1)) #(y.PATIENT_ID.isin(Yy.PATIENT_ID)) & (y)
    # assert False
    table1[chosen_model]=[
                        f'{truepositives} ({truepositives*100/len(Xx):2.2f} %)',
                        f'{100*Xx.FEMALE.sum()/len(Xx):2.2f} %',
                        f'{a1} ({a1*100/len(Xx):2.2f} %) ',
                        f'{a2} ({a2*100/len(Xx):2.2f} %) ',
                        f'{a3} ({a3*100/len(Xx):2.2f} %) ',
                        f'{a4} ({a4*100/len(Xx):2.2f} %) ',
                        f'{a5} ({a5*100/len(Xx):2.2f} %) ',
                        f'{a85plus} ({a85plus*100/len(Xx):2.2f} %) ']+comorb
    
    
    # table1=pd.DataFrame({chosen_model:table1})
    if chosen_model=='neuralNetworkRandom_43':
        table=table1
    else:
        table=table.join(table1,lsuffix=' - MLP', rsuffix=' - LR')
print(table.style.to_latex())
    
#%%
