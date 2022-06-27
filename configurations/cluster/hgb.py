#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
import importlib
import numpy as np
import os
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

parser = argparse.ArgumentParser(description='Train HGB algorithm and save model')
parser.add_argument('chosen_config', type=str,
                    help='The name of the config file (without .py), which must be located in configurations/cluster.')
parser.add_argument('experiment',
                    help='The name of the experiment config file (without .py), which must be located in configurations.')
parser.add_argument('--seed-hparam', metavar='seed',type=int, default=argparse.SUPPRESS,
                    help='Random seed for hyperparameter tuning')
parser.add_argument('--seed-sampling', metavar='seed',type=int, default=argparse.SUPPRESS,
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

if  args.experiment!=experiment:#required arg, will always be there
    EXPERIMENT=args.experiment #OVERRIDE (this is the only variable from the imported experiment module that needs to be changed, because it creates model and prediction directories)
MODELPATH=MODELSPATH+EXPERIMENT+'/'
USEDCONFIGPATH+=EXPERIMENT+'/'
ALGORITHM='HGB'
CONFIGNAME='configHGB.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
METRICSPATH=os.path.join(METRICSPATH,EXPERIMENT)
TRACEBACK=False

""" SETTINGS FOR THE RANDOM FOREST """
seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else SEED
   
FOREST=HistGradientBoostingClassifier(loss='auto', max_bins=255,
                                   # categorical_features=cat,
                                   monotonic_cst=None,
                                   warm_start=False, early_stopping=False,
                                   scoring='loss', validation_fraction=0.1,
                                   n_iter_no_change=10, tol=1e-07,
                                   random_state=seed_hparam)


learning_rate=[0.01,0.1,0.3,0.5,0.5,1.0]
max_iter=[500,1000,1500,2500,5000] #number of trees
max_leaf_nodes=[15,30,45,60,None]
max_depth=[None, 10,20,30,50]
min_samples_leaf=[10,20,40,80]
l2_regularization=[0.0,0.1,0.5,1.0]

RANDOM_GRID = {'learning_rate':learning_rate,
               'max_iter':max_iter,
               'max_leaf_nodes':max_leaf_nodes,
               'max_depth':max_depth,
               'min_samples_leaf':min_samples_leaf,
               'l2_regularization':l2_regularization}


N_ITER=50
CV=3
