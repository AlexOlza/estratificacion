#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:28:50 2023

@author: aolza
"""
import sys
import os
import configurations.utility as util
from python_settings import settings as config
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()

logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import joblib
from dataManipulation.dataPreparation import getData
import pandas as pd
import re

logistic_modelpath=config.ROOTPATH+'models/urgcmsCCS_parsimonious/'
linear_modelpath=config.ROOTPATH+'models/costCCS_parsimonious/'

logistic_modelname='logistic20230324_111354'
linear_modelname='linear20230324_130625'

logistic_predpath=re.sub('models','predictions',logistic_modelpath)
linear_predpath=re.sub('models','predictions',linear_modelpath)

logistic_global_model=joblib.load(logistic_modelpath+f'{logistic_modelname}.joblib')
linear_global_model=joblib.load(linear_modelpath+f'{linear_modelname}.joblib')
logistic_sameprev_model=joblib.load(logistic_modelpath+'logistic_gender_balanced.joblib')


log_global_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_global_model.feature_names_in_, logistic_global_model.coef_[0])},
                                        orient='index')
lin_global_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(linear_global_model.feature_names_in_, linear_global_model.coef_[0])},
                                        orient='index')
log_sameprev_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_sameprev_model.feature_names_in_, logistic_sameprev_model.coef_[0])},
                                        orient='index')
print('logistic global female coef:',log_global_coefs.loc['FEMALE'].values )
print('logistic same prevalence female coef:',log_sameprev_coefs.loc['FEMALE'].values )
print('linear global female coef:', lin_global_coefs.loc['FEMALE'].values  )


"""
HOSPITALIZATION BIG TABLE
"""
def concat_preds(file1,file2):
    muj=pd.read_csv(file1)
    muj['FEMALE']=1
    hom=pd.read_csv(file2)
    hom['FEMALE']=0
    return pd.concat([muj,hom])
import numpy as np
def table(path, modelname):
    # Global model
    global_preds=concat_preds(path+f'{modelname}_Mujeres_calibrated_2018.csv',
                              path+f'{modelname}_Hombres_calibrated_2018.csv')   
    # Separate models
    modeltype=re.sub('[0-9]|_','',modelname) #this string will contain either "logistic" or "linear"
    separate_preds=concat_preds(path+f'{modeltype}Mujeres_calibrated_2018.csv',
                                path+f'{modeltype}Hombres_calibrated_2018.csv')    
    if modeltype=='logistic':
        # Same prevalence model
        sameprevalence_preds=concat_preds(path+'logistic_gender_balanced_Mujeres_calibrated_2018.csv',
                                          path+'logistic_gender_balanced_Hombres_calibrated_2018.csv')
        
        # Overall metrics: auc and brier
        from sklearn.metrics import roc_auc_score,brier_score_loss 
        def AUC_muj(x): return roc_auc_score(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def AUC_hom(x): return roc_auc_score(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        def brier_muj(x): return brier_score_loss(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def brier_hom(x): return brier_score_loss(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        
        overall_metrics=[[AUC_muj(x),AUC_hom(x),brier_muj(x),brier_hom(x)] for x in [global_preds,separate_preds,sameprevalence_preds]]
    else:
        
        #Overall metrics: R2 and RMSE
        
        pd.DataFrame(overall_metrics,columns=['AUC_women','AUC_men','brier_women','brier_men'],index=['global','separate','same_prevalence'])
        