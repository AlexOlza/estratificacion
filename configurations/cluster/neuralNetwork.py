#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:22:32 2022

@author: aolza
"""
"""EXTERNAL IMPORTS"""
import keras_tuner as kt
from keras import layers, initializers
from keras.callbacks import EarlyStopping
from keras import initializers
from tensorflow import keras
import re
import argparse
import importlib
import numpy as np
import os
#%%
"""REPRODUCIBILITY"""
seed_value=42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value) # tensorflow 2.x
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=3,allow_soft_placement=True)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
#%%
""" OUR ARGUMENTS AND IMPORTS """
parser = argparse.ArgumentParser(description='Train HGB algorithm and save model')
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
args = parser.parse_args()
experiment='configurations.'+re.sub('hyperparameter_|undersampling_|full_|variability_|fixsample_','',args.experiment)

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
ALGORITHM='neuralNetwork'
CONFIGNAME='neuralNetwork.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
TRACEBACK=False

seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else SEED

from main.cluster.clr_callback import CyclicLR
""" DEFINITION OF THE HYPERPARAMETER SPACE """
def clr(RANDOM_GRID):
    	return CyclicLR(
    	mode='triangular',
    	base_lr=RANDOM_GRID['low'],
    	max_lr=RANDOM_GRID['high'],
    	step_size=  RANDOM_GRID['step'])


class HYPERMODEL( kt.HyperModel ):
    # def __init__(self, grid):
    #     self.RANDOM_GRID = grid
    def build(self, RANDOM_GRID):
        #neurons in input layer
        RANDOM_GRID.Int("units_0", min_value=32, max_value=1024, step=32)
        #number of hidden layers
        RANDOM_GRID.Int('n_hidden', min_value=0, max_value=3, step=1)
        #neurons in each hidden layer: inside build function
        #learning rate
        RANDOM_GRID.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        #activation function
        RANDOM_GRID.Choice('activation', ['relu','elu'])
        #callbacks and training details
        RANDOM_GRID.Boolean('CyclicLR')
        RANDOM_GRID.Float('low',min_value=1e-4,max_value=5e-3)
        RANDOM_GRID.Float('high',min_value=1e-2,max_value=5e-2)
        RANDOM_GRID.Fixed('EarlyStopping', True)
        RANDOM_GRID.Boolean("shuffle")
        
        self.CALLBACKS = [EarlyStopping(monitor='val_loss',mode='min',
                                      patience=5,
                                      restore_best_weights=True)]
        if RANDOM_GRID['CyclicLR']:
            self.CALLBACKS.append(clr(RANDOM_GRID))
        # BUILD MODEL
        model = keras.Sequential()
        #input layer
        model.add(
            layers.Dense(
                units=RANDOM_GRID['units_0'],
                activation=RANDOM_GRID['activation'],
            )
        )
        #hidden layers
        for i in range(1,RANDOM_GRID["n_hidden"]+1):
            model.add(
            layers.Dense(
                # Tune number of units separately.
                units=RANDOM_GRID.Int(f"units_{i}", min_value=32, max_value=1024, step=32),
                activation=RANDOM_GRID['activation']))
        #output layer
        model.add(layers.Dense(1, activation='sigmoid',name='output',
                kernel_initializer=initializers.he_uniform(seed=seed_hparam)))

        if RANDOM_GRID['CyclicLR']:
            model.compile(
            optimizer=keras.optimizers.Adam(), 
               loss="binary_crossentropy", metrics=[keras.metrics.AUC(),
                                                                      keras.metrics.Recall(top_k=20000),
                                                                      keras.metrics.Precision(top_k=20000)])
        else:
            model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=RANDOM_GRID['lr']), 
               loss="binary_crossentropy", metrics=[keras.metrics.AUC(),
                                                                  keras.metrics.Recall(top_k=20000),
                                                                  keras.metrics.Precision(top_k=20000)])
       
        return model

    def fit(self, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=self.RANDOM_GRID['shuffle'],
            callbacks=self.CALLBACKS
            **kwargs,
        )

