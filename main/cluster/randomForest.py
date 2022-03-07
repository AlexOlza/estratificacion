#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:19:44 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import argparse

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
n_iter= args.n_iter if hasattr(args, 'n_iter') else config.N_ITER
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

np.random.seed(seed_sampling)

pred16,y17=getData(2016)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'
print('Sample size ',len(pred16))

#%%
"""BALANCEO LAS CLASES Y TRAIN-TEST SPLIT"""
idx=np.array(list(itertools.chain.from_iterable((y17[config.COLUMNS] >= 1).values)))
ing_indices = y17.loc[idx].index
noing_indices = y17.loc[(~idx)].index
half_sample_size = len(ing_indices)  # Equivalent to len(data[data.Healthy == 0])
random_indices = np.random.choice(noing_indices, half_sample_size, 
                                  replace=False)

X=pd.concat([pred16.loc[random_indices],pred16.loc[ing_indices]])
na_indices=X[X.isna().any(axis=1)].index
X.drop(na_indices,axis=0,inplace=True)
print('first IDs (X)')
print(X.head().PATIENT_ID)
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

y=pd.concat([y17.loc[random_indices],y17.loc[ing_indices]])
y.drop(na_indices,axis=0,inplace=True)
print('first IDs (y)')
print(y.head().PATIENT_ID)
y=np.where(y[config.COLUMNS]>=1,1,0)
y=y.ravel()
print('Dropping NA')
print('Sample size ',len(X))

if not isinstance(config.FOREST,RandomForestClassifier):
    print('exec happening')
    exec("estimator="+config.FOREST)
else:
    print('exec not needed')
    estimator=config.FOREST
np.random.seed(seed_hparam)
forest = RandomizedSearchCV(estimator = estimator, 
                               param_distributions = config.RANDOM_GRID,
                               n_iter = n_iter,
                               cv = config.CV, 
                               verbose=0,
                               random_state=seed_hparam,
                               n_jobs =-1)
forest.fit(X,y)
#%%
""" SAVE TRAINED MODEL """
if hasattr(args, 'model_name'):
    util.savemodel(config,forest.best_estimator_,name=args.model_name)
else:
    util.savemodel(config,forest.best_estimator_) 
#
