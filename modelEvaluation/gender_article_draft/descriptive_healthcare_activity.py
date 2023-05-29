#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:09:31 2023

@author: alex
"""
import sys
import os
import configurations.utility as util
from python_settings import settings as config
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
import pandas as pd
importlib.invalidate_caches()

logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import numpy as np

X,y=getData(2017, predictors='PATIENT_ID|FEMALE|AGE_[0-9]+$', CCS=False, PHARMACY=False)

X['AGE_older_than_85']=np.where(X.filter(regex='AGE').sum(axis=1)==0,1,0)
X['AGE_younger_than_54']=X['AGE_0004']+X['AGE_0511']+X['AGE_1217']+X['AGE_1834']+X['AGE_3544']+X['AGE_4554']
X=X.drop(['AGE_0004','AGE_0511','AGE_1217','AGE_1834','AGE_3544','AGE_4554'],axis=1)
X['AGE']=X.filter(regex='AGE_',axis=1).idxmax(axis=1)

diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2017]),
                  usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE', 'PLACE'],
                  # nrows=100,
                  index_col=False)

diags=pd.merge(diags,X[['PATIENT_ID','FEMALE','AGE']],on='PATIENT_ID', how='right')


#%%
for value, df in diags.groupby('FEMALE'):
    gender= 'Females' if value==1 else 'Males'
    print(f'{len(df)*100/len(diags)} % of diagnostic codes are for {gender}')

n_diags={'Males':[0]*len(diags.loc[(diags.FEMALE==0) & (diags.CIE_CODE.isna())]),
         'Females':[0]*len(diags.loc[(diags.FEMALE==1) & (diags.CIE_CODE.isna())])}
for value, df in diags.loc[~ diags.CIE_CODE.isna()].groupby(['FEMALE','PATIENT_ID']):
    gender= 'Females' if value[0]==1 else 'Males'
    n_diags[gender].append(len(df))
    
n_diags_females=pd.DataFrame(n_diags['Females'],columns=['N'])
n_diags_males=pd.DataFrame(n_diags['Males'],columns=['N'])

df=n_diags_females.describe()
df.rename(columns={'N':'Females'},inplace=True)
df['Males']=n_diags_males.describe()
print(df.to_markdown())

#%%
n_diags={}
for age in diags.AGE.unique():
    n_diags[f'Males_{age}']=[0]*len(diags.loc[(diags.FEMALE==0) & (diags.CIE_CODE.isna()) & ( diags.AGE==age)])
    n_diags[ f'Females_{age}'] = [0]*len(diags.loc[(diags.FEMALE==1) & (diags.CIE_CODE.isna()) & ( diags.AGE==age)])

for value, df in diags.loc[~ diags.CIE_CODE.isna()].groupby(['FEMALE', 'AGE','PATIENT_ID']):
    genderage= 'Females_'+value[1] if value[0]==1 else 'Males_'+value[1]
    n_diags[genderage].append(len(df))
    
#%%
df=pd.DataFrame()
for key in n_diags.keys():
    df[key]=pd.DataFrame(n_diags[key]).describe()