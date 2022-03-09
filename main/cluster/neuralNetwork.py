#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://keras.io/guides/keras_tuner/getting_started/
Created on Tue Mar  8 10:29:07 2022

@author: aolza
"""
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow import keras
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import argparse

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

args = parser.parse_args()

chosen_config='configurations.cluster.'+args.chosen_config
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 
# from configurations.cluster import configRandomForest as randomForest_settings
import configurations.utility as util
util.makeAllPaths()
seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else config.SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else config.SEED
# n_iter= args.n_iter sif hasattr(args, 'n_iter') else config.N_ITER
model_name=config.ROOTPATH+'neuraltest'
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
np.random.seed(seed_sampling)

X,y=getData(2016,predictors=r'FEMALE|AGE_')
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.99, random_state=42)
print('Sample size ',len(y_train))

#%% 
""" FIT MODEL """
np.random.seed(seed_hparam)
# import tensorflow as tf
# class MyTuner(kt.Tuner):
#   def run_trial(self,trial, *args, **kwargs):
#     patience = hp.Int('patience', 0, 3, default=1)
#     callbacks = tf.keras.callbacks.ReduceLROnPlateau(patience=patience)
#     super(MyTuner, self).run_trial(*args, **kwargs, callbacks=callbacks)


tuner = config.MyTuner(X_train, y_train,X_test, y_test,
                     objective='val_loss',
                     # max_epochs=10,
                     # factor=3,
                     directory=config.ROOTPATH+'my_dir',
                     project_name='intro_to_kt')

tuner.search(epochs=10)

# print('Best score obtained: {0}'.format(rs_keras.best_score_))
# print('Parameters:')
# for param, value in rs_keras.best_params_.items():
#     print('\t{}: {}'.format(param, value))

#%%
""" SAVE TRAINED MODEL """
# cv_results_df = pd.DataFrame(grid_result.cv_results_)
# cv_results_df.to_csv('gridsearch.csv')
# print(grid_result.best_params_)

# hypermodel = config.HYPERMODEL()
# best_hp = tuner.get_best_hyperparameters()[0]
# model = hypermodel.build(best_hp)
# print('Fitting best model with the whole data-set')
# hypermodel.fit(best_hp, model, X, y, epochs=1)

# hypermodel.save(model_name)
