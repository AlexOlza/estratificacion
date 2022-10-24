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
name= args.model_name if hasattr(args,'model_name') else config.ALGORITHM
epochs=args.seed_hparam if hasattr(args, 'epochs') else 500
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData

X,y=getData(2016)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'


try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass
y=y[config.COLUMNS]



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42)
X_test, X_test2, y_test, y_test2 = train_test_split( X_test, y_test, test_size=0.5, random_state=42)

print('Sample size ',len(y_train))
print('---------------------------------------------------'*5)
#%% 
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

hist, bin_edges = np.histogram(y_train, bins = 50)
classes = y_train.apply(lambda x: pd.cut(x, bin_edges, labels = False, 
                                                  include_lowest = True)).values
sample_weights = compute_sample_weight('balanced', classes)
#%%
""" FIT MODEL """
np.random.seed(seed_hparam)
print('Seed ', seed_hparam)

model_name=config.MODELPATH+name

print('Tuner: Random')
tuner= kt.BayesianOptimization(config.build_model,
                     objective=kt.Objective("val_loss", direction="min"),
                      max_trials=15,
                      overwrite=True,
                      num_initial_points=4,
                     directory=model_name+'_search',
                     project_name=name,
                     )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(X_train, y_train,epochs=5, validation_split=0.2,callbacks=[stop_early],
             # sample_weight=sample_weights
             )
print('---------------------------------------------------'*5)
print('SEARCH SPACE SUMMARY:')
print(tuner.search_space_summary())  

#%%
""" SAVE TRAINED MODEL """
""" work in progress """
best_hp = tuner.get_best_hyperparameters()[0]

model = tuner.hypermodel.build(best_hp)
history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.2,callbacks=[stop_early],
                    sample_weight=pd.Series(sample_weights)
                    )

val_acc_per_epoch = history.history['val_root_mean_squared_error']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


#%%
