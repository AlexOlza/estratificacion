#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:51:45 2022

@author: alex
"""

import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

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

import joblib
import os
from dataManipulation.dataPreparation import getData
import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoCV
#%%
models=['lasso_only_gma','lasso_full']
coefs={}
for available_lasso in Path(config.MODELPATH).glob('lasso*.joblib'):
    modelname=str(available_lasso).split('/')[-1]
    print('loading ',modelname)
    model=joblib.load(available_lasso)
    coefs[modelname]={name:[coef] for name, coef in zip(model.feature_names_in_,model.coef_)}

#%%
if not all([Path(os.path.join(config.MODELPATH,f'{m}.joblib')).is_file() for m in models]):
    indices=["GMA_num_patol",
           "GMA_num_sist","GMA_peso-ip",
           "GMA_riesgo-muerte"]
    patologias=["GMA_dm","GMA_ic","GMA_epoc",
           "GMA_hta","GMA_depre","GMA_vih","GMA_c_isq","GMA_acv",
           "GMA_irc","GMA_cirros",
           "GMA_osteopor","GMA_artrosis",
           "GMA_artritis","GMA_demencia","GMA_dolor_cron"]
    additional_columns=indices+patologias
    np.random.seed(config.SEED)
    X,y=getData(2016,
                 CCS=True,
                 PHARMACY=True,
                 BINARIZE_CCS=True,
                 GMA=True,
                 GMACATEGORIES=True,
                 GMA_DROP_DIGITS=0,
                 additional_columns=additional_columns)
    from dataManipulation.dataPreparation import reverse_one_hot
    X=reverse_one_hot(X)
    y=y[config.COLUMNS]
    print('Sample size ',len(X))
#%%
if not Path(os.path.join(config.MODELPATH,'lasso_only_gma.joblib')).is_file() :
    logistic=LassoCV(n_alphas=30,verbose=1, n_jobs=-1)#lasso
    
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
    fit=logistic.fit(X.filter(regex='GMA'), y)
    print('fitting time: ',time()-t0)
    modelname, modelfilename=util.savemodel(config, fit, name='lasso_only_gma', return_=True)
    print(modelname, modelfilename)

#%%
if not Path(os.path.join(config.MODELPATH,'lasso_full.joblib')).is_file() :
    logistic=LassoCV(n_alphas=30,verbose=1, n_jobs=-1)#lasso
    
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
    fit=logistic.fit(X, y)
    print('fitting time: ',time()-t0)
    
    fullcoefs={name:coef for name, coef in zip(fit.feature_names_in_,fit.coef_)}
    
    modelname, modelfilename=util.savemodel(config, fit, name='lasso_full', return_=True)
    print(modelname, modelfilename)
#%%
plot=False
if plot:
    import pandas as pd
    for k,v in coefs.items():
        df=pd.DataFrame(v).T
        df.rename(columns={0:'beta'},inplace=True)
        df.loc[df.beta!=0].plot.bar(title=k)
       
