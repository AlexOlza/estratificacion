#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:32:34 2022

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
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%%

z=X['FEMALE']
Xx=X.drop('FEMALE', axis=1)
print('Sample size ',len(X), 'positive: ',sum(z))
assert not 'AGE_85GT' in X.columns

#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

to_drop=['PATIENT_ID','ingresoUrg']
for c in to_drop:
    try:
        X.drop(c,axis=1,inplace=True)
        util.vprint('dropping col ',c)
    except:
        pass
        util.vprint('pass')
from time import time
t0=time()
# fit=logistic.fit(Xx, z)
# print('fitting time: ',time()-t0)
from sklearn.metrics import roc_auc_score
# print('Logistic AUC=',roc_auc_score(z.ravel(), fit.predict_proba(Xx)[:,0]))
#%%
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import argparse
import os
import importlib
import random

import pandas as pd
import numpy as np 
from pathlib import Path
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


args, unknown_args = parser.parse_known_args()

chosen_config='configurations.cluster.'+args.chosen_config
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 

import configurations.utility as util
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import predict
from modelEvaluation.compare import performance

seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else config.SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else config.SEED
name= args.model_name if hasattr(args,'model_name') else config.ALGORITHM
epochs=args.seed_hparam if hasattr(args, 'epochs') else 500

#%%
# X_train, X_test, y_train, y_test = train_test_split( Xx, z, test_size=0.2, random_state=42)
# X_test, X_test2, y_test, y_test2 = train_test_split( X_test, y_test, test_size=0.5, random_state=42)


# print('Sample size ',len(y_train))
print('---------------------------------------------------'*5)
 

""" FIT MODEL """
np.random.seed(seed_hparam)
print('Seed ', seed_hparam)

model_name=config.MODELPATH+'propensity_score_model'

print('Tuner: Bayesian')
tuner= kt.BayesianOptimization(config.build_model,
                     objective=kt.Objective("val_loss", direction="min"),
                      max_trials=10,
                      overwrite=False,
                      num_initial_points=4,
                     directory=model_name+'_search',
                     project_name='propensity_score_model',   
                     
                     )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(Xx, z,epochs=10, validation_split=0.3,callbacks=[stop_early],
              )
print('---------------------------------------------------'*5)
print('SEARCH SPACE SUMMARY:')
print(tuner.search_space_summary())  


""" SAVE TRAINED MODEL """
""" work in progress """
best_hp = tuner.get_best_hyperparameters()[0]


model = tuner.hypermodel.build(best_hp)
history = model.fit(Xx, z, epochs=50, verbose=1, validation_split=0.3,callbacks=[stop_early],
                    )


val_acc_per_epoch = history.history['loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

keras.models.save_model(model, os.path.join(config.MODELPATH,'propensity_score_model'))
print('Saved ',os.path.join(config.MODELPATH,'propensity_score_model'))