#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:23:21 2021

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
from sklearn.linear_model import RidgeCV

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%%

y=y[config.COLUMNS]
print('Sample size ',len(X))

#%%
linear=RidgeCV(alphas=np.logspace(start=1e-4, stop=100, num=10),scoring='neg_mean_square_error')

to_drop=['PATIENT_ID','ingresoUrg', 'AGE_85GT']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')
from time import time
t0=time()
fit=linear.fit(X, y)
print('fitting time: ',time()-t0)
#%%
modelname, modelfilename=util.savemodel(config, fit, return_=True)

#%%
plot=False
if plot:
    import pandas as pd
    df=pd.DataFrame({X.columns[i]:fit.coef_[0][i] for i in range(len(X.columns))})
    df.rename(columns={0:'beta'},inplace=True)
    df.nlargest(10,'beta').plot.bar(title=modelname)
    df.nsmallest(10,'beta').plot.bar(title=modelname)
