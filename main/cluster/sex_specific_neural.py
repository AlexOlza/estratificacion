#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN TWO SEPARATE NEURAL NETWORK MODELS FOR MALES AND FEMALES
Created on Fri Mar 18 12:50:05 2022

@author: aolza
"""
import sys
import os
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow import keras
import tensorflow as tf
import argparse
import re
import importlib
import random
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
#%%
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
parser = argparse.ArgumentParser(description='Train and save sex specific Neural Networks')
parser.add_argument('chosen_config', type=str,
                    help='The name of the config file (without .py), which must be located in configurations/cluster.')
parser.add_argument('experiment',
                    help='The name of the experiment config file (without .py), which must be located in configurations.')

parser.add_argument('--n-iter', metavar='n_iter',type=int, default=argparse.SUPPRESS,
                    help='Number of iterations for the random grid search (hyperparameter tuning)')


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
model_name= args.model_name if hasattr(args,'model_name') else config.ALGORITHM 
epochs=args.seed_hparam if hasattr(args, 'epochs') else 500
#%%
np.random.seed(config.SEED)

X,y=getData(2016) 
#%%
female=X['FEMALE']==1
male=X['FEMALE']==0

sex=[ 'Hombres', 'Mujeres']
np.random.seed(seed_hparam)
#%%
for group, groupname in zip([male,female],sex):
    print(groupname)
    model_name_group=model_name+groupname
    Xgroup=X.loc[group]
    ygroup=y.loc[group]

    assert (all(Xgroup['FEMALE']==1) or all(Xgroup['FEMALE']==0))
    
    print('Sample size ',len(Xgroup))
    if config.ALGORITHM=='neuralNetwork':
        ygroup=np.where(ygroup[config.COLUMNS]>=1,1,0)
        ygroup=ygroup.ravel()
    elif config.ALGORITHM=='neuralRegression':
        ygroup=ygroup[config.COLUMNS].to_numpy()

    else:
        assert False, 'This script is only suitable for neural networks. Check your configuration!'
    to_drop=['PATIENT_ID','ingresoUrg', 'FEMALE']
    for c in to_drop:
        try:
            Xgroup.drop(c,axis=1,inplace=True)
            util.vprint('dropping col ',c)
        except:
            pass
            util.vprint('pass')
    from time import time
    X_train, X_test, y_train, y_test = train_test_split( Xgroup, ygroup, test_size=0.2, random_state=42)
    X_test, X_test2, y_test, y_test2 = train_test_split( X_test, y_test, test_size=0.5, random_state=42)

    print('Sample size ',len(y_train))
    t0=time()
    
    print('Tuner: Random')
    tuner = config.MyRandomTuner(X_train, y_train.reshape(-1,1),X_test, y_test.reshape(-1,1),
                 objective=kt.Objective("val_loss", direction="min"),
                 max_trials=10, 
                overwrite=True,
                 seed=seed_hparam,
                 cyclic=False,
                 directory=config.MODELPATH+model_name_group+'_search',
                 project_name=model_name)
      
    tuner.search(epochs=30)
    print('---------------------------------------------------'*5)
    print('SEARCH SPACE SUMMARY:')
    print(tuner.search_space_summary())  
    print('fitting time: ',time()-t0)

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
    
    
    print('Best hyperparameters:')
    print(best_hp_)
    print('---------------------------------------------------'*5)
    print(f'Retraining ({epochs} epochs):')
    t0=time()
    config.keras_code(X_train,y_train,X_test2,y_test2, epochs=epochs,**best_hp_,
                callbacks=callbacks, save=True, saving_path=config.MODELPATH+model_name_group, verbose=1)
    util.saveconfig(config,config.USEDCONFIGPATH+model_name.split('/')[-1]+'.json')
    print('Saved ')
    print('retraining time: ',time()-t0)
