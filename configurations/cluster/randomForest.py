#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
import importlib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

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
ALGORITHM='randomForest'
CONFIGNAME='configRandomForest.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
TRACEBACK=False

""" SETTINGS FOR THE RANDOM FOREST """
IMPORTS=["from sklearn.ensemble import RandomForestClassifier","from sklearn.model_selection import RandomizedSearchCV"]

seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else SEED

FOREST=RandomForestClassifier(criterion='gini',
                              min_weight_fraction_leaf=0.0, 
                              max_leaf_nodes=None, 
                              min_impurity_decrease=0.0, 
                              min_impurity_split=None,
                              bootstrap=True, 
                              oob_score=True, 
                              n_jobs=-1,
                              random_state=seed_hparam)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 30, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
#Maximum proportion of samples to build each tree:
# max_samples=[0.3,0.5,0.7]
# Create the random grid
RANDOM_GRID = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

N_ITER=50
CV=3
