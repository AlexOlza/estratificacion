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

logistic_modelpath=config.ROOTPATH+'/models/urgcmsCCS_parsimonious/'
linear_modelpath=config.ROOTPATH+'/models/costCCS_parsimonious/'

logistic_global_model=joblib.load(logistic_modelpath+'logistic20230324_111354.joblib')
linear_global_model=joblib.load(linear_modelpath+'linear20230324_130625.joblib')
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


