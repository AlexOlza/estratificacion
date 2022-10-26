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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from  tensorflow.keras import backend as K  
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

util.makeAllPaths()
seed_sampling= args.seed_sampling if hasattr(args, 'seed_sampling') else config.SEED #imported from default configuration
seed_hparam= args.seed_hparam if hasattr(args, 'seed_hparam') else config.SEED
name= args.model_name if hasattr(args,'model_name') else config.ALGORITHM
epochs=args.seed_hparam if hasattr(args, 'epochs') else 500
#%%
""" BEGGINNING """

X,y=getData(2016)
X=X[[c for c in X if X[c].max()>0]]
print(config.PREDICTORREGEX)



variables={'Demo':'PATIENT_ID|FEMALE|AGE_[0-9]+$',
           'DemoDiag':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_' if 'ACG' in config.PREDICTORREGEX else 'PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS',
           'DemoDiagPharma':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_|RXMG_' if 'ACG' in config.PREDICTORREGEX else None,
           'DemoDiagPharmaIsomorb':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_(?!NUR11|RES10)|RXMG_(?!ZZZX000)|ACG_' if 'ACG' in config.PREDICTORREGEX else None
           }

assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'

PATIENT_ID=X.PATIENT_ID
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
    
except:
    pass

if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
    compute_weight=True
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
    compute_weight=False
 
#%%

#%%
for key, val in variables.items():
    print('STARTING ',key, val)
    if not val:
        continue
    if Path(os.path.join(config.MODELPATH,key)).is_dir(): #the model is already there
        continue
    Xx=X.filter(regex=val, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split( Xx, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_test2, y_test, y_test2 = train_test_split( X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    
    if compute_weight:
        from sklearn.utils import class_weight as cw
        class_weight = cw.compute_class_weight('balanced',
                                                     classes=np.unique(y_train),
                                                     y=y_train)
        class_weight={i:class_weight[i] for i in range(len(class_weight))}
    else:
        class_weight=None
        
    print(class_weight)
    print('Sample size ',len(y_train))
    print('---------------------------------------------------'*5)
 

    """ FIT MODEL """
    np.random.seed(seed_hparam)
    print('Seed ', seed_hparam)
    
    model_name=config.MODELPATH+key

    print('Tuner: Bayesian')
    tuner= kt.BayesianOptimization(config.build_model,
                         objective=kt.Objective("val_loss", direction="min"),
                          max_trials=10,
                          overwrite=False,
                          num_initial_points=4,
                         directory=model_name+'_search',
                         project_name=key,   
                         
                         )
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    
    tuner.search(X_train, y_train,epochs=10, validation_split=0.2,callbacks=[stop_early],
                 class_weight=class_weight
                 # sample_weight=sample_weights
                 )
    print('---------------------------------------------------'*5)
    print('SEARCH SPACE SUMMARY:')
    print(tuner.search_space_summary())  
    
    
    """ SAVE TRAINED MODEL """
    """ work in progress """
    best_hp = tuner.get_best_hyperparameters()[0]
    
    model = tuner.hypermodel.build(best_hp)
    history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.2,callbacks=[stop_early],
                        )
    
    
    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    keras.models.save_model(model, os.path.join(config.MODELPATH,key))
    print('Saved ',os.path.join(config.MODELPATH,key))
#%%


X, y=getData(2017)
X=X[[c for c in X if X[c].max()>0]]
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
    compute_weight=True
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
   
# X=pd.concat([X, PATIENT_ID], axis=1) if not 'PATIENT_ID' in X else X
# y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y
#%%
table=pd.DataFrame()
for key, val in variables.items():
    if not val:
        continue
    probs,auc=predict(key,experiment_name=config.EXPERIMENT,year=2018,
                    X=X.filter(regex=val, axis=1), y=y)
    recall, ppv, _, _ = performance(obs=probs.OBS, pred=probs.PRED, K=20000)
    brier=brier_score_loss(y_true=probs.OBS, y_prob=probs.PRED)
    ap=average_precision_score(probs.OBS,probs.PRED)
    table=pd.concat([table,
                     pd.DataFrame.from_dict({'Model':[key], 'AUC':[ auc], 'AP':[ap],
                      'R@20k': [recall], 'PPV@20K':[ppv], 
                      'Brier':[brier]})])
    
#%%
print(table.to_markdown(index=False))