#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:51:54 2022

@author: aolza
https://github.com/slundberg/shap
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import argparse
import os
import re
import importlib
# import shap
from python_settings import settings as config
from pathlib import Path
import pkg_resources
pkg_resources.require("Shap==0.40")
import shap
import importlib
importlib.reload(shap)

#%%
if not config.configured: 
    import configurations.utility as util
    experiment=input('Experiment...')
    print('Available models:')
    p = Path(os.path.join(os.environ['USEDCONFIG_PATH'],experiment)).glob('**/*.json')
    files = [x.stem for x in p if x.is_file()]
    print(files)
    model=input('Model...')
    config_used=os.path.join(os.environ['USEDCONFIG_PATH'],experiment,model+'.json')
    configuration=util.configure(config_used,TRACEBACK=True, VERBOSE=True)

# if not config.configured:
#     chosen_config='configurations.cluster.'+args.chosen_config
#     importlib.invalidate_caches()
#     settings=importlib.import_module(chosen_config,package='estratificacion')
#     config.configure(settings) # configure() receives a python module
assert config.configured 

import configurations.utility as util
from dataManipulation.dataPreparation import getData

from modelEvaluation.detect import detect_models
util.makeAllPaths()

#%%


X,y=getData(2017)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
y[config.COLUMNS]=np.where(y[config.COLUMNS]>=1,1,0)
Xx=X.drop('PATIENT_ID',axis=1)
#%%
available_models=detect_models()
for i, m in enumerate(available_models):
    print(i, m)
i = int(input('Choose the number of the desired model: '))
model=keras.models.load_model(config.MODELPATH+available_models[i])
#%%

#%%
sample=shap.sample(Xx, 10)
explainer = shap.KernelExplainer(data=sample,model=model, 
                                 output_names=list(Xx.columns),
                                 max_evals=600)

shap_values = explainer.shap_values(sample)#Xx.iloc[:100,:].to_numpy()
explanation=shap.Explanation(shap_values[0],feature_names=list(Xx.columns))
#%%
#these work
shap.summary_plot(shap_values, Xx)
shap.plots.beeswarm(explanation)
shap.plots.beeswarm(explanation,order=explanation.abs.max(0))
shap.plots.bar(explanation,max_display=30)
#%%
explanation.data=sample
sex = ["Women" if explanation[i,"FEMALE"].data == 1 else "Men" for i in range(explanation.shape[0])]
shap.plots.bar(explanation.cohorts(sex).abs.mean(0))
#%%
shap.plots.bar(explanation.cohorts(2).abs.mean(0))
#%%
# shap.plots.beeswarm(explainer.expected_value)
sum_shap=[sum(s) for s in shap_values[0]]

forcemax=shap.plots.force(explainer.expected_value[0], shap_values[0][np.argmax(sum_shap)],ordering_keys='reverse',feature_names=list(Xx.columns),out_names=['neg','pos'],matplotlib=True)
forcemin=shap.plots.force(explainer.expected_value[0], shap_values[0][np.argmin(sum_shap)],ordering_keys='reverse',feature_names=list(Xx.columns),out_names=['neg','pos'],matplotlib=True)

shap.plots.force(explanation)
shap.plots.scatter(explanation)
#%%
shap.plots.bar(explanation[0])
