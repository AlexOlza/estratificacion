#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:15:43 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
year=int(sys.argv[3])

import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
import re
import os
import numpy as np
import pandas as pd
from dataManipulation.dataPreparation import getData, generateCCSData
#%%

X,y=getData(year)
#%%
n,p={},{}
for c in X.drop('PATIENT_ID',axis=1):
    n[c]= X[c].sum()
    
#%%

for c in X.drop('PATIENT_ID',axis=1):
    p[c]= sum(np.where(X[c]>=1,1,0))/len(X)

#%%
from collections import Counter
K=15
c = Counter(n)
cp= Counter(p)
most_common = c.most_common(K) 
most_prevalent= cp.most_common(K) 

#%%
#Build dataframe with descriptions
ccs=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str, usecols=['CCS CATEGORY', 'CCS CATEGORY DESCRIPTION'])
ccs.drop_duplicates(inplace=True)
#%%
def to_df(List, ccsDF):
    keys=[re.sub('CCS','',e[0]) for e in List]
    numbers=[e[1] for e in List]
    df=pd.DataFrame({'CCS CATEGORY': keys, 'N':numbers})
    df=pd.merge(df, ccsDF, how='left')
    return df

print('MOST COMMON CCSs:')
mc=to_df(most_common, ccs)
print(mc)
mc.to_excel(f'most_common_CCSs_{year}.xlsx',index=False)

print('MOST PREVALENT CCSs:')
mp=to_df(most_prevalent, ccs)
print(mp)
mp.to_excel(f'most_prevalent_CCSs_{year}.xlsx',index=False)