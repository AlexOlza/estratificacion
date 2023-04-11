#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:50:36 2023

@author: aolza
"""
""" MODELS WITH MEDIAN RECALL_20000"""
logistic_modelname='logistic20220207_122835'
neural_modelname='neuralNetworkRandom_43'
randomforest_modelname='randomForest_59'
hgb_modelname='hgb_61'


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
import modelEvaluation.calibrate as cal
import pandas as pd
import numpy as np
import re
from comparative_article_plotting_functions import *

import shap
import joblib
from tensorflow import keras

X,y=getData(2016)
assert False
#%%
""" RANDOM FOREST VARIABLE IMPORTANCE PLOTS"""

rf_model=joblib.load(config.ROOTPATH+f'models/hyperparameter_variability_urgcms_excl_nbinj/{randomforest_modelname}.joblib')

importances = rf_model.feature_importances_
forest_importances = pd.Series(importances, index=X.drop('PATIENT_ID',axis=1).columns.values)
forest_importances.loc['AGE']=forest_importances.filter(regex='AGE').sum()
forest_importances.drop(forest_importances.filter(regex='AGE_').index,axis=0,inplace=True)
fig, ax = plt.subplots()
forest_importances.nlargest(20).plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig(config.FIGUREPATH+'/comparative/variable_importance_random_forest.jpeg',dpi=300)
#%%
""" HGB VARIABLE IMPORTANCE PLOTS"""

hgb_model=joblib.load(config.ROOTPATH+f'models/hyperparameter_variability_urgcms_excl_nbinj/{hgb_modelname}.joblib')

Xx= X.sample(1000).drop('PATIENT_ID',axis=1)
explainer = shap.TreeExplainer(hgb_model)
shap_values = explainer.shap_values(Xx)
# shap.plots.bar(shap_values)

shap.summary_plot(shap_values,Xx)

from sklearn.preprocessing import OrdinalEncoder
Xxcat=Xx.copy()
# Xxcat['AGE_85']=np.where(Xxcat.filter(regex='AGE_').sum(axis=1)==0,1,0)
Xxcat['AGE']=Xxcat.filter(regex='AGE_').idxmax(axis=1)
Xxcat.drop(Xxcat.filter(regex='AGE_'),axis=1,inplace=True)
Xxcat['AGE']=OrdinalEncoder().fit_transform(Xxcat['AGE'].to_numpy().reshape(-1, 1)).ravel()

shap_df=pd.DataFrame(shap_values,columns=Xx.columns.values)
shap_df['AGE']=shap_df.filter(regex='AGE_').sum(axis=1)
shap_df.drop(shap_df.filter(regex='AGE_').columns,axis=1,inplace=True)

shap_val_cat=shap_df.to_numpy()

fig, ax = plt.subplots()
fig=shap.summary_plot(shap_df.to_numpy(),Xxcat,show=False)#,feature_names=shap_df.columns)
plt.savefig(config.FIGUREPATH+'/comparative/variable_importance_hgb_summary.jpeg',dpi=300)


fig, ax = plt.subplots()
shap_df.abs().mean().nlargest(20).plot.bar(ax=ax)
ax.set_title("Feature importances using shap values")
ax.set_ylabel("mean(abs(shap values))")
fig.tight_layout()
plt.savefig(config.FIGUREPATH+'/comparative/variable_importance_hgb_bars.jpeg',dpi=300)

#%%
Xx= Xx.sample(500)
neural_model=keras.models.load_model(config.ROOTPATH+f'models/hyperparameter_variability_urgcms_excl_nbinj/{neural_modelname}')

explainer = shap.KernelExplainer(neural_model,data=Xx, link='logit')
shap_values = explainer.shap_values(Xx)
shap_values=np.array(shap_values).reshape(np.array(shap_values).shape[1:])

Xxcat=Xx.copy()
# Xxcat['AGE_85']=np.where(Xxcat.filter(regex='AGE_').sum(axis=1)==0,1,0)
Xxcat['AGE']=Xxcat.filter(regex='AGE_').idxmax(axis=1)
Xxcat.drop(Xxcat.filter(regex='AGE_'),axis=1,inplace=True)
Xxcat['AGE']=OrdinalEncoder().fit_transform(Xxcat['AGE'].to_numpy().reshape(-1, 1)).ravel()

shap_df=pd.DataFrame(shap_values,columns=Xx.columns.values)
shap_df['AGE']=shap_df.filter(regex='AGE_').sum(axis=1)
shap_df.drop(shap_df.filter(regex='AGE_').columns,axis=1,inplace=True)

shap.summary_plot(shap_df.to_numpy(),Xxcat)#,feature_names=shap_df.columns)
plt.savefig(config.FIGUREPATH+'/comparative/variable_importance_neural_summary.jpeg',dpi=300)


fig, ax = plt.subplots()
shap_df.abs().mean().nlargest(20).plot.bar(ax=ax)
ax.set_title("Feature importances using shap values")
ax.set_ylabel("mean(abs(shap values))")
fig.tight_layout()
plt.savefig(config.FIGUREPATH+'/comparative/variable_importance_neural_bars.jpeg',dpi=300)
