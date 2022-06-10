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
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import modelEvaluation.calibrate as cal
import pandas as pd
import numpy as np
import re
from comparative_article_plotting_functions import *
#%%

#%%
yr=2018
X,y=getData(2017)
X16,y17=getData(2016)
#%%

#%%
""" MATERIALS AND METHODS: Comments on variability assessment"""
K=20000
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
for violin in (True, False):
    for together in (True, False):
        boxplots(metrics, violin, together)
""" BRIER BOXPLOTS """
brier_boxplot_zoom(metrics) #violins
brier_boxplot_zoom(metrics, False) #boxplots
#%%
""" CALIBRATION: RELIABILITY DIAGRAMS """

median_models={}
for metric in ['Brier', 'Brier Before']:
    mediandf=metrics.groupby(['Algorithm'])[metric].agg([ median]).stack(level=0)
    for alg in metrics.Algorithm.unique():
        if alg=='logistic':
            continue
        df_alg=metrics.loc[metrics.Algorithm==alg].to_dict(orient='list')
        perc50=mediandf.loc[alg]['median']
        chosen_model=list(df_alg['Model'])[list(df_alg[metric]).index(perc50)]
        print(metrics.loc[metrics.Model==chosen_model][metric])
        predpath=re.sub(config.EXPERIMENT,'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)
        
        try:
            median_models[metric][model_labels([chosen_model])[0]]= cal.calibrate(chosen_model, yr,
                                                           experiment_name='hyperparameter_variability_urgcms_excl_nbinj',
                                                           filename=os.path.join(predpath,f'{chosen_model}_calibrated_{yr}.csv'))
        except KeyError:
            median_models[metric]={model_labels([chosen_model])[0]: cal.calibrate(chosen_model,yr,   experiment_name='hyperparameter_variability_urgcms_excl_nbinj',
                                                           filename=os.path.join(predpath,f'{chosen_model}_calibrated_{yr}.csv'))}


median_models['Brier']['LR']= cal.calibrate(logistic_model,yr)
median_models['Brier Before']['LR']= cal.calibrate(logistic_model,yr)

cal.plot(median_models['Brier'],consistency_bars=False)
cal.plot(median_models['Brier Before'],consistency_bars=False)

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
        
#%%
# m1,m2,m3,m4=risk_groups.keys()
# disj_union=set(risk_groups[m1]).symmetric_difference(set(risk_groups[m2])).symmetric_difference(set(risk_groups[m3])).symmetric_difference(set(risk_groups[m4]))

# simdif=len(disj_union)

# print('The 4 risk groups identified contained...')
# print(simdif, '(simetric difference)')
# print(f'... patients not in common, that is, {simdif*100/20000:2.2f} %')
# print(f'That means {20000-simdif} patients were selected by the four algorithms')
#%%
""" TABLE 1:  DEMOGRAPHICS"""
# chosen_model='neuralNetworkRandom_43'
table=pd.DataFrame()
for chosen_model in ['neuralNetworkRandom_43', logistic_model]:
    Xx=X.loc[X.PATIENT_ID.isin(risk_groups[chosen_model])]
    Yy=y.loc[y.PATIENT_ID.isin(risk_groups[chosen_model])]
    female=Xx['FEMALE']==1
    male=Xx['FEMALE']==0
    sex=[ 'Women','Men']
    
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
    
    table1= pd.DataFrame(index=['N in 2017 (%)', 'Hospitalized in 2018', 
                                'Aged 0-17',
                                'Aged 18-64',
                                'Aged 65-69',
                                'Aged 70-79',
                                'Aged 80-84',
                                'Aged 85+']+list(comorbidities.keys()))
    comorb={}
    for group, groupname in zip([female,male],sex):
        print(groupname)
        Xgroup=Xx.loc[group]
        ygroup=Yy.loc[group]
        comorb[groupname]=[]
        for disease, EDClist in comorbidities.items():
            s=[Xgroup[EDC].sum() for EDC in EDClist]
            comorb[groupname].append(f'{sum(s)} ({sum(s)*100/len(Xgroup):2.2f} %)')
            print(disease, 'total (M+W): ',sum([Xx[EDC].sum() for EDC in EDClist]))
        # ygroup18=y.loc[group18]
        a1=sum(Xgroup.AGE_0004)+sum(Xgroup.AGE_0511)+sum(Xgroup.AGE_0511)
        a2=sum(Xgroup.AGE_1834)+sum(Xgroup.AGE_3544)+sum(Xgroup.AGE_4554)+sum(Xgroup.AGE_5564)
        a3=sum(Xgroup.AGE_6569)
        a4=sum(Xgroup.AGE_7074)+sum(Xgroup.AGE_7579)
        a5=sum(Xgroup.AGE_8084)
        a85plus=len(Xgroup)-(a1+a2+a3+a4+a5)
        positives=sum(np.where(ygroup.urgcms>=1,1,0))
        table1[groupname]=[f'{len(Xgroup)} ({len(Xgroup)*100/len(Xx):2.2f} %)',
                            f'{positives} ({positives*100/len(Xgroup):2.2f} %)',
                            f'{a1} ({a1*100/len(Xgroup):2.2f} %) ',
                            f'{a2} ({a2*100/len(Xgroup):2.2f} %) ',
                            f'{a3} ({a3*100/len(Xgroup):2.2f} %) ',
                            f'{a4} ({a4*100/len(Xgroup):2.2f} %) ',
                            f'{a5} ({a5*100/len(Xgroup):2.2f} %) ',
                            f'{a85plus} ({a85plus*100/len(Xgroup):2.2f} %) ']+comorb[groupname]
        
        
        
    if chosen_model=='neuralNetworkRandom_43':
        table=table1
    else:
        table=table.join(table1,lsuffix=' - MLP', rsuffix=' - LR')
print(table.style.to_latex())
    
#%%
