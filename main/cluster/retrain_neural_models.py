#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:32:03 2022

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
model_name= f'neuralNetwork_{seed_hparam}' 
epochs=args.epochs if hasattr(args, 'epochs') else 100
cyclic=args.cyclic
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData

X,y=getData(2016)
X_test,y_test=getData(2017)

assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
y_test=np.where(y_test[config.COLUMNS]>=1,1,0)
y_test=y_test.ravel()
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
    X_test.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

#this is in fact just undersampling
_, X_test, _, y_test = train_test_split( X_test, y_test, test_size=0.2, random_state=42)

print('---------------------------------------------------'*5)

#%%
""" LOAD MODEL """
model=keras.models.load_model(config.MODELPATH+model_name)
print(f'Retraining ({epochs} epochs):')
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',
                                      patience=5,
                                      restore_best_weights=True)]
history=model.fit(X, y, callbacks=callbacks, epochs=epochs,
                   validation_data=(X_test,y_test), verbose=1)
keras.models.save_model(model, filepath=config.MODELPATH+model_name+'_ret')
# util.saveconfig(config,config.USEDCONFIGPATH+model_name.split('/')[-1]+'.json')
# print('Saved ')