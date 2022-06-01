#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS CONFIGURATION IS INTENDED TO HAVE
REGULARIZATION AND
MORE VARIABILITY DURING HYPERPARAMETER TUNING
Created on Tue May 31 11:53:18 2022

@author: aolza
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
parser.add_argument('--random_tuner','-r', dest='random_tuner',action='store_true', default=False,
                    help='Use random grid search (default: False, use Bayesian)')
parser.add_argument('--clr','-c', dest='cyclic',action='store_true', default=False,
                    help='Use Cyclic Learning Rate')
args, unknown = parser.parse_known_args()
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
ALGORITHM='neuralRegularized'
CONFIGNAME='neuralRegularized.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
FIGUREPATH=os.path.join(ROOTPATH,'figures',EXPERIMENT)

TRACEBACK=False

seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else SEED
model_name= args.model_name if hasattr(args,'model_name') else 'neural'
from main.cluster.clr_callback import CyclicLR
""" DEFINITION OF THE HYPERPARAMETER SPACE """
def clr(low, high, step):
    	return CyclicLR(
    	mode='triangular2',
    	base_lr=low,
    	max_lr=high,
    	step_size= step)

def build_model(units_0, n_hidden, activ, cyclic, early, **kwargs):
    lr=kwargs.get('lr',None)
    # low=kwargs.get('low',None)
    # high=kwargs.get('high',None)
    hidden_units=kwargs.get('hidden_units',{})
    # BUILD MODEL
    model = keras.Sequential()
    #input layer
    model.add(
        keras.layers.Dense(
            units=units_0,
            activation=activ,
            activity_regularizer='l2'
        )
    )
    #hidden layers
    for i in range(1,n_hidden+1):
        model.add(
        keras.layers.Dense(
            # Tune number of units separately.
            units=hidden_units[f"units_{i}"],
            activity_regularizer='l1_l2',
            activation=activ))
    #output layer
    model.add(keras.layers.Dense(1, activation='sigmoid',name='output',
            kernel_initializer=keras.initializers.he_uniform(seed=seed_hparam)))

    if cyclic:
        model.compile(
        optimizer=keras.optimizers.Adam(), 
           loss="binary_crossentropy", metrics=[keras.metrics.AUC(),
                                                                  keras.metrics.Recall(),
                                                                  keras.metrics.Precision()])
    else:
        model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
           loss="binary_crossentropy", metrics=[keras.metrics.AUC(),keras.metrics.BinaryCrossentropy()])
   
    
    return model

def keras_code(x_train, y_train, x_val, y_val,
               units_0, n_hidden, activ, cyclic, early,  callbacks,
               save=False,
               saving_path=None,
               epochs=1,
               **kwargs
               ):
    verbose=kwargs.get('verbose',0)
    lr=kwargs.get('lr',None)
    batch_size=kwargs.get('batch_size', 256)
    hidden_units=kwargs.get('hidden_units',{})
    # Build model
    model = build_model(units_0, n_hidden, activ, cyclic, early, 
                        lr=lr, hidden_units=hidden_units)
    callbacks.append(keras.callbacks.Callback())
    # Train & eval model
    history=model.fit(x_train, y_train, callbacks=callbacks, epochs=epochs,
                      batch_size=batch_size, validation_data=(x_val,y_val), verbose=verbose)
    
    if save and saving_path:
        # Save model
        model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    # y_pred = model.predict(x_val)
    # 
    print(history.history)
    return({'val_loss':history.history['val_loss'][-1] }) 

def run(tuner, trial, **kwargs):
        cyclic=tuner.cyclic
        hp = trial.hyperparameters
        #batch size
        batch_size = hp.Int('batch_size', 64, 1024, step=32)
        #neurons in input layer
        units_0=hp.Int("units_0", min_value=312, max_value=1024, step=32)
        #number of hidden layers
        n_hidden=hp.Int('n_hidden', min_value=0, max_value=10, step=2)
        units={}

        for i in range(1,n_hidden+1):
            # Tune number of units separately.
            units[f"units_{i}"]=hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32)
        cyclic=hp.Fixed('cyclic', cyclic)
        with hp.conditional_scope('cyclic', False):
            #learning rate
            lr=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="linear")
        with hp.conditional_scope('cyclic', True):
             low=hp.Float('low',min_value=1e-4,max_value=5e-3, sampling="linear")
             high=hp.Float('high',min_value=6e-3,max_value=5e-2, sampling="linear")
        #activation function
        activ=hp.Choice('activ', ['relu','elu'])
        #callbacks and training details
        
       
        early=hp.Fixed('early', True)
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',
                                      patience=15,
                                      restore_best_weights=True)]
        if cyclic:
            callbacks.append(clr(low, high, step=len(tuner.y_train // 512)))
        print(units_0, n_hidden, activ, cyclic, early, callbacks,
            units, lr)
        return keras_code(tuner.x_train, tuner.y_train, tuner.x_val, tuner.y_val,
            units_0, n_hidden, activ, cyclic, early, callbacks,
            hidden_units=units, lr=lr, batch_size=batch_size )


class MyRandomTuner(kt.RandomSearch):
    def __init__(self, x_train, y_train, x_val, y_val, cyclic=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.x_train=x_train
        self.y_train=y_train
        self.x_val=x_val
        self.y_val=y_val
        self.cyclic=cyclic

    def run_trial(self, trial, **kwargs):
        return(run(self, trial, **kwargs))
    
class MyBayesianTuner(kt.BayesianOptimization):
    def __init__(self, x_train, y_train, x_val, y_val, cyclic=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.x_train=x_train
        self.y_train=y_train
        self.x_val=x_val
        self.y_val=y_val
        self.cyclic=cyclic
        
    def run_trial(self, trial, **kwargs):
        return(run(self, trial, **kwargs))
# tuner = MyTuner(
#     max_trials=3, overwrite=True, directory="my_dir", project_name="keep_code_separate",
# )
# tuner.search()
# # Retraining the model
# best_hp = tuner.get_best_hyperparameters()[0]
# keras_code(**best_hp.values, saving_path="/tmp/best_model")