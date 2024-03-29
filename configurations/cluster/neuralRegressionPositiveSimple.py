#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:22:32 2022

@author: alex
"""
"""EXTERNAL IMPORTS"""
import keras_tuner as kt
from tensorflow import keras
import re
import argparse
import importlib
import numpy as np
import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
"""REPRODUCIBILITY"""
seed_value=42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value) # tensorflow 2.x
# 5. Configure a new global `tensorflow` session
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=3,allow_soft_placement=True)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
#%%
""" OUR ARGUMENTS AND IMPORTS """
parser = argparse.ArgumentParser(description='Train neural regression and save model')
parser.add_argument('chosen_config', type=str,
                    help='The name of the config file (without .py), which must be located in configurations/cluster.')
parser.add_argument('experiment',
                    help='The name of the experiment config file (without .py), which must be located in configurations.')

parser.add_argument('--model-name', metavar='model_name',type=str, default=argparse.SUPPRESS,
                    help='Custom model name to save (provide without extension nor directory)')

args, unknown = parser.parse_known_args()
experiment='configurations.'+args.experiment
importlib.invalidate_caches()

"""THIS EMULATES 'from experiment import *' USING IMPORTLIB 
info: 
    https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
"""
mdl=importlib.import_module(experiment,package='estratificacion')
# is there an __all__?  if so respect it
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    # otherwise we import all names that don't begin with _
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
globals().update({k: getattr(mdl, k) for k in names}) #brings everything into namespace

from configurations.default import *

if  args.experiment!=experiment:#required arg
    EXPERIMENT=args.experiment #OVERRIDE (this is the only variable from the imported experiment module that needs to be changed, because it creates moddel and prediction directories)
MODELPATH=MODELSPATH+EXPERIMENT+'/'
USEDCONFIGPATH+=EXPERIMENT+'/'
ALGORITHM='neuralRegressionPositiveSimple'
CONFIGNAME='neuralRegressionPositiveSimple.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
FIGUREPATH=os.path.join(ROOTPATH,'figures',EXPERIMENT)
METRICSPATH=os.path.join(METRICSPATH,EXPERIMENT)

TRACEBACK=False

seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else SEED
model_name= args.model_name if hasattr(args,'model_name') else ALGORITHM

""" DEFINITION OF THE HYPERPARAMETER SPACE """

from tensorflow.keras import backend as K   
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model(hp):
    
    # BUILD MODEL
    model = keras.Sequential()
    #input layer
   
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    n_hidden= hp.Int('n_hidden', min_value=1, max_value=3, step=1)
    hidden_units={f"units_{i}":hp.Int(f"units_{i}", min_value=32, max_value=512, step=32) for i in range(1,n_hidden+1)}
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    #hidden layers
    for i in range(1,n_hidden+1):
        model.add(
        keras.layers.Dense(
            # Tune number of units separately.
            units=hidden_units[f"units_{i}"],
            activation='elu'))
    #output layer
    model.add(keras.layers.Dense(1, activation='relu',name='output',
            kernel_initializer=keras.initializers.he_uniform(seed=42)))

   
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate ), 
       loss="mean_squared_error",metrics=[coeff_determination,keras.metrics.RootMeanSquaredError()])
   
    
    return model

