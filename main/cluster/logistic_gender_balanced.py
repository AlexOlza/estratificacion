#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:59:43 2022

@author: aolza
"""

import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%% 
def gender_balanced_undersampling(X,y):
    import itertools
    XFemale=X.loc[X.FEMALE==1]
    yFemale=y.loc[y.PATIENT_ID.isin(XFemale.PATIENT_ID)]
    XMale=X.loc[X.FEMALE==0]
    yMale=y.loc[y.PATIENT_ID.isin(XMale.PATIENT_ID)]

    admittedFemale=yFemale.loc[(yFemale[config.COLUMNS]>=1).values.ravel()].PATIENT_ID
    healthyFemale=yFemale.loc[(yFemale[config.COLUMNS]==0).values.ravel()].PATIENT_ID
    half_sample_size = len(admittedFemale)
    random_healthyFemale = np.random.choice(healthyFemale, half_sample_size, 
                                  replace=False)
    
      
    XFemaleBalanced=pd.concat([XFemale.loc[XFemale.PATIENT_ID.isin(random_healthyFemale)],XFemale.loc[XFemale.PATIENT_ID.isin(admittedFemale)]])
    yFemaleBalanced=pd.concat([yFemale.loc[yFemale.PATIENT_ID.isin(random_healthyFemale)],yFemale.loc[yFemale.PATIENT_ID.isin(admittedFemale)]])

    
    admittedMale=yMale.loc[(yMale[config.COLUMNS]>=1).values.ravel()].PATIENT_ID
    healthyMale=yMale.loc[(yMale[config.COLUMNS]==0).values.ravel()].PATIENT_ID

    random_healthyMale = np.random.choice(healthyMale, half_sample_size, 
                                  replace=False)
    random_admittedMale = np.random.choice(admittedMale, half_sample_size, 
                                  replace=False)
    
      
    XMaleBalanced=pd.concat([XMale.loc[XMale.PATIENT_ID.isin(random_healthyMale)],XMale.loc[XMale.PATIENT_ID.isin(random_admittedMale)]])
    yMaleBalanced=pd.concat([yMale.loc[yMale.PATIENT_ID.isin(random_healthyMale)],yMale.loc[yMale.PATIENT_ID.isin(random_admittedMale)]])

 
    
    #this would be 0.5 always in the current implementation
    prevalence_Females=sum(np.where(yFemaleBalanced[config.COLUMNS]>=1,1,0))/len(yFemaleBalanced)
    prevalence_Males=sum(np.where(yMaleBalanced[config.COLUMNS]>=1,1,0))/len(yMaleBalanced)
    
    assert sum(np.where(yMaleBalanced[config.COLUMNS]>=1,1,0))[0]==len(XMaleBalanced)/2, 'Hospitalizations incorrectly balanced for males'
    assert sum(np.where(yFemaleBalanced[config.COLUMNS]>=1,1,0))[0]==len(XFemaleBalanced)/2, 'Hospitalizations incorrectly balanced for females'
    assert len(XMaleBalanced)==len(XFemaleBalanced), 'Gender unbalanced'
    assert prevalence_Females==prevalence_Males, 'Unequeal prevalences'
    
    return(XFemaleBalanced,yFemaleBalanced,XMaleBalanced,yMaleBalanced,prevalence_Females)

Xf,yf,Xm,ym,prev_f=gender_balanced_undersampling(X,y)

X=pd.concat([Xf,Xm])
y=pd.concat([yf,ym])
#%%
print('first IDs (X)')
print(X.head().PATIENT_ID)
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

print('first IDs (y)')
print(y.head().PATIENT_ID)
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Sample size ',len(X), 'positive: ',sum(y))

#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')
from time import time
t0=time()
fit=logistic.fit(X, y)
print('fitting time: ',time()-t0)
#%%
util.savemodel(config, fit,name='logistic_gender_balanced')

