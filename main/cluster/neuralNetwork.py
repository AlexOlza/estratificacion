#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://keras.io/guides/keras_tuner/getting_started/
Created on Tue Mar  8 10:29:07 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow import keras
import tensorflow as tf
import argparse
import os
import re
import importlib
import random
#%%
"""REPRODUCIBILITY"""
seed_value=42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value) # tensorflow 2.x
parser = argparse.ArgumentParser(description='Train and save Neural Network')
parser.add_argument('chosen_config', type=str,
                    help='The name of the config file (without .py), which must be located in configurations/cluster.')
parser.add_argument('experiment',
                    help='The name of the experiment config file (without .py), which must be located in configurations.')
parser.add_argument('--seed-hparam', metavar='seed_hparam',type=int, default=argparse.SUPPRESS,
                    help='Random seed for hyperparameter tuning')
parser.add_argument('--seed-sampling', metavar='seed_sampling',type=int, default=argparse.SUPPRESS,
                    help='Random seed for undersampling')
parser.add_argument('--model-name', metavar='model_name',type=str, default=argparse.SUPPRESS,
                    help='Custom model name to save (provide without extension nor directory)')
parser.add_argument('--n-iter', metavar='n_iter',type=int, default=argparse.SUPPRESS,
                    help='Number of iterations for the random grid search (hyperparameter tuning)')
parser.add_argument('--random_tuner','-r', dest='random_tuner',action='store_true', default=False,
                    help='Use random grid search (default: False, use Bayesian)')
parser.add_argument('--clr','-c', dest='cyclic',action='store_true', default=False,
                    help='Use Cyclic Learning Rate')

args = parser.parse_args()

chosen_config='configurations.cluster.'+args.chosen_config
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 

import configurations.utility as util
util.makeAllPaths()
seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else config.SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else config.SEED
model_name= args.model_name if hasattr(args,'model_name') else 'neuralNetwork' 
cyclic=args.cyclic
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData

X,y=getData(2016)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print('Sample size ',len(y_train))
print('---------------------------------------------------'*5)
#%% 
""" FIT MODEL """
np.random.seed(seed_hparam)
print('Seed ', seed_hparam)

if not args.random_tuner:
    name=re.sub('neuralNetwork','neuralNetworkBayesian',model_name)
    name=re.sub('neuralNetworkBayesian','neuralNetworkBayesianCLR',name) if cyclic else name
    model_name=config.MODELPATH+name
    print('Tuner: BayesianOptimization')
    tuner = config.MyBayesianTuner(X_train, y_train.reshape(-1,1),X_test, y_test.reshape(-1,1),
                     objective=kt.Objective("val_loss", direction="min"),
                     max_trials=100,
                     overwrite=True,
                     num_initial_points=4,
                     seed=seed_hparam,
                     cyclic=cyclic,
                     directory=model_name+'_search',
                     project_name=name)
else:
    name=re.sub('neuralNetwork','neuralNetworkRandom',model_name)
    name=re.sub('neuralNetworkRandom','neuralNetworkRandomCLR',name) if cyclic else name
    model_name=config.MODELPATH+name
    print('Tuner: Random')
    tuner = config.MyRandomTuner(X_train, y_train.reshape(-1,1),X_test, y_test.reshape(-1,1),
                 objective=kt.Objective("val_loss", direction="min"),
                 max_trials=5, 
                 overwrite=True,
                 seed=seed_hparam,
                 cyclic=cyclic,
                 directory=model_name+'_search',
                 project_name=name)
  
tuner.search(epochs=30)
print('---------------------------------------------------'*5)
print('SEARCH SPACE SUMMARY:')
print(tuner.search_space_summary())  

#%%
""" SAVE TRAINED MODEL """
""" work in progress """
best_hp = tuner.get_best_hyperparameters()[0]
# try:
#     print('units: ',best_hp.values['units'])
# except KeyError:
#     best_hp.values['units']=[]
best_hp_={k:v for k,v in best_hp.values.items() if not k.startswith('units')}
best_hp_['units_0']=best_hp.values['units_0']
best_hp_['hidden_units']={f'units_{i}':best_hp.values[f'units_{i}'] for i in range(1,best_hp.values['n_hidden']+1)}
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',
                                      patience=5,
                                      restore_best_weights=True)]
if cyclic:
    callbacks.append(config.clr(best_hp.values['low'], best_hp.values['high'], step=(len(y) // 1024)))


print('Best hyperparameters:')
print(best_hp_)
print('---------------------------------------------------'*5)
print('Retraining (100 epochs):')
config.keras_code(X,y,X_train,y_train, epochs=100,**best_hp_,
                  callbacks=callbacks, save=True, saving_path=model_name, verbose=2)
util.saveconfig(config,config.USEDCONFIGPATH+model_name.split('/')[-1]+'.json')
print('Saved ')

#%%
""" FUTURE IDEAS:
    EXPLAINABILITY: https://github.com/slundberg/shap"""
model=keras.models.load_model(model_name)
import shap
explainer = shap.KernelExplainer(data=shap.sample(X_test, 100),model=model, 
                                 output_names=list(X.columns),
                                 max_evals=600)

shap_values = explainer.shap_values(X_test.iloc[:1000,:].to_numpy())

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0,:], matplotlib=True)

shap.summary_plot(shap_values[0], X_test, plot_type="bar")
# shap.plots.beeswarm(explainer.expected_value)
force=shap.plots.force(explainer.expected_value[0], shap_values[0][0],ordering_keys='reverse',feature_names=list(X.columns),out_names=['neg','pos'],matplotlib=True)
"""
PROB INGRESO 0.05
X_test.iloc[1,:].CCS49
Out[111]: 0 no diabetes

X_test.iloc[1,:].CCS159
Out[112]: 2 urinary system diseases

X_test.iloc[1,:].CCS94
Out[113]: 1 ear conditions

X_test.iloc[1,:].CCS198
Out[114]: 1 Other inflammatory condition of skin 

X_test.iloc[1,:].CCS174
Out[115]: 1
"""
force=shap.plots.force(explainer.expected_value[0], shap_values[0][44],ordering_keys='reverse',feature_names=list(X.columns),out_names=['neg','pos'],matplotlib=True)
"""
PROB INGRESO 0.31

X_test.iloc[44,:].CCS105
Out[118]: 5 Diseases of the heart

X_test.iloc[44,:].CCS131
Out[119]: 1 ,Respiratory failure; insufficiency; arrest (adult) [

X_test.iloc[44,:].CCS51
Out[120]: 6 Other endocrine disorders 

X_test.iloc[44,:].CCS133
Out[121]: 1  Other lower respiratory disease 

X_test.iloc[44,:].CCS98
Out[122]: 1

X_test.iloc[44,:].CCS259
Out[123]: 1

X_test.iloc[44,:].CCS130
Out[124]: 1
"""
# allpoints=shap.plots.force(explainer.expected_value,shap_values[0])
# shap.plots.scatter(shap_values[:,1], color=shap_values[0])
# # shap.plots.beeswarm(shap_values)
# import lime
# from lime.lime_tabular import LimeTabularExplainer
#%%
# limeexplainer = LimeTabularExplainer(X_test.iloc[:1000,:].to_numpy().reshape(1,-1),
                                                      
#                                                    feature_names=list(X.columns),discretize_continuous=False)
# #%%
# import numpy as np
# i = np.random.randint(0, X_test.shape[0])
# #%%
# def predict_proba(x):
#     p=model.predict(x)[0][0]
#     print(np.array([1-p, p]).reshape(1,-1).ravel())
#     return np.array([1-p, p]).reshape(1,-1).ravel()
# exp = limeexplainer.explain_instance(X_test.iloc[i,:].to_numpy().reshape(1,-1), 
#                                      predict_proba)