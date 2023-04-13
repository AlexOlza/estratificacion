#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:52:06 2023

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
from modelEvaluation.compare import compare,performance
import pandas as pd
import numpy as np
import re
import zipfile
#%%

#%%
yr=2018
X,y=getData(2017)
X16,y17=getData(2016)
#%%
""" MATERIALS AND METHODS: Comments on variability assessment"""
K=0.1
recall_list,ppv_list=[],[]
names=[]
zipfilename=config.ROOTPATH+'/predictions/hyperparameter_variability_urgcms_excl_nbinj.zip'
zfile=zipfile.ZipFile(zipfilename,'r')
""" RESULTS. TABLE 2 """
for predfile in zfile.namelist():
    if not '_calibrated_2018' in predfile:
        continue
    if ('CLR' in predfile) or ('Bayesian' in predfile):
        continue
    preds=pd.read_csv(zfile.open(predfile))
    recall, ppv,_,_ =performance(preds.OBS,preds.PRED,K=K)
    recall_list.append(recall)
    ppv_list.append(ppv)
    names.append(predfile)
preds_log=pd.read_csv(config.PREDPATH+f'/{logistic_model}_calibrated_2018.csv')
recall, ppv,_,_ =performance(preds_log.OBS,preds_log.PRED,K=K)
recall_list.append(recall)
ppv_list.append(ppv)
names.append('logistic')
df = pd.DataFrame(list(zip(names,recall_list,ppv_list)),
                  columns=['Model', f'Recall_{K}',f'PPV_{K}'])
print(df.to_markdown(index=False, ))

df['Algorithm'] = [re.sub('/|calibrated|.csv|hyperparameter_variability_urgcms_excl_nbinj|_|[0-9]', '', model) 
                   for model in df['Model'].values]
df=df.loc[df.Algorithm.isin(['logistic','hgb','randomForest','neuralNetworkRandom'])]
# Option 1: Median --- (Interquartile range)
table2=pd.DataFrame()
from scipy.stats import iqr
for metric in [f'Recall_{K}', f'PPV_{K}']:
    median=df.groupby(['Algorithm'])[metric].median()
    IQR=df.groupby(['Algorithm'])[metric].agg(iqr)
    table2[metric]=[f'{m:1.3f} ({i:.2E})' for m, i in zip(median.values, IQR.values)]
    table2.index=IQR.index
  
print(table2.style.to_latex())

higher_better={'Score': True, 'Recall_{K}': True,
               'PPV_20000': True, 'Brier':False, 'AP':True}
def use_f_3(x):
    return "%.4f" % x
def use_E(x):
    return "%.2e" % x
# Option 2: Subtables with descriptive
table2=pd.DataFrame()
for metric in [f'Recall_{K}', f'PPV_{K}']:
    table=df.groupby(['Algorithm'])[metric].describe()[['25%','50%', '75%','std']].sort_values('50%', ascending=[False])
    table2=pd.concat([table2,table])
print(table2.to_latex(formatters=[ use_f_3, use_f_3, use_f_3, use_E]))
print('\n'*2)

def median(x):
    return x.quantile(0.5,interpolation='nearest')

"""
K=0.05

\begin{tabular}{lrrrr}
\toprule
{} &    25\% &    50\% &    75\% &      std \\
Algorithm           &        &        &        &          \\
\midrule
neuralNetworkRandom & 0.2856 & 0.2858 & 0.2861 & 4.20e-04 \\
hgb                 & 0.2817 & 0.2821 & 0.2828 & 7.64e-04 \\
logistic            & 0.2815 & 0.2815 & 0.2815 & 0.00e+00 \\
randomForest        & 0.2590 & 0.2651 & 0.2655 & 4.02e-03 \\
neuralNetworkRandom & 0.3242 & 0.3244 & 0.3248 & 4.76e-04 \\
hgb                 & 0.3197 & 0.3201 & 0.3210 & 8.67e-04 \\
logistic            & 0.3195 & 0.3195 & 0.3195 & 0.00e+00 \\
randomForest        & 0.2940 & 0.3009 & 0.3013 & 4.56e-03 \\
\bottomrule
\end{tabular}

k=0.10
\begin{tabular}{lrrrr}
\toprule
{} &    25\% &    50\% &    75\% &      std \\
Algorithm           &        &        &        &          \\
\midrule
neuralNetworkRandom & 0.4377 & 0.4380 & 0.4384 & 5.41e-04 \\
hgb                 & 0.4349 & 0.4353 & 0.4355 & 6.60e-04 \\
logistic            & 0.4340 & 0.4340 & 0.4340 & 0.00e+00 \\
randomForest        & 0.4042 & 0.4135 & 0.4136 & 5.72e-03 \\
neuralNetworkRandom & 0.2484 & 0.2486 & 0.2488 & 3.07e-04 \\
hgb                 & 0.2468 & 0.2470 & 0.2472 & 3.75e-04 \\
logistic            & 0.2463 & 0.2463 & 0.2463 & 0.00e+00 \\
randomForest        & 0.2294 & 0.2346 & 0.2347 & 3.25e-03 \\
\bottomrule
\end{tabular}
"""