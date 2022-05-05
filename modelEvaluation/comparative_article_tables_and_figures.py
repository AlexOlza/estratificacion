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
import pandas as pd
import numpy as np
import re
#%%

X,y=getData(2017)
X16,y17=getData(2016)
#%%
""" TABLE 1:  DEMOGRAPHICS"""
female=X['FEMALE']==1
male=X['FEMALE']==0
sex=[ 'Women','Men']
# table1={'Women':pd.DataFrame(), 'Men':pd.DataFrame()}
table1= pd.DataFrame(index=['N (%)', 'Hospitalized in 2017', 
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

table2=pd.DataFrame()
from scipy.stats import iqr
for metric in ['Score', 'Recall_20000', 'PPV_20000', 'F1_20000','Brier']:
    median=metrics.groupby(['Algorithm'])[metric].median()
    IQR=metrics.groupby(['Algorithm'])[metric].agg(iqr)
    table2[metric]=[f'{m:1.3f} ({i:.2E})' for m, i in zip(median.values, IQR.values)]
    table2.index=IQR.index
  
print(table2.to_latex())
