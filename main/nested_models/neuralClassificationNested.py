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
parser.add_argument('--tuner', metavar='tuner',type=int, default=argparse.SUPPRESS,
                    help='Type of tuner (pass an int for random, omit for bayesian)')

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
epochs=args.epochs if hasattr(args, 'epochs') else 500
tuner=args.tuner if hasattr(args, 'tuner') else 'bayesian'
#%%
print('KerasTuner version: ', kt.__version__)

""" BEGGINNING """

X,y=getData(2016)
X=X[[c for c in X if X[c].max()>0]]
print(config.PREDICTORREGEX)

undersampling=True if 'highcost' in config.EXPERIMENT else False


if (not 'ACG' in config.PREDICTORREGEX):
    if (hasattr(config, 'PHARMACY')):
        CCSPHARMA='PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS|PHARMA' if config.PHARMACY else None
    else: CCSPHARMA= None
    if (hasattr(config, 'GMACATEGORIES')):
        CCSGMA='PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS|PHARMA|GMA' if config.GMACATEGORIES else None
    else: CCSGMA= None
else: 
    CCSPHARMA=None
    CCSGMA=None

variables={'Demo':'PATIENT_ID|FEMALE|AGE_[0-9]+$',
           'DemoDiag':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_' if 'ACG' in config.PREDICTORREGEX else 'PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS',
           'DemoDiagPharma':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_|RXMG_' if 'ACG' in config.PREDICTORREGEX else CCSPHARMA,
           'DemoDiagPharmaIsomorb':'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_(?!NUR11|RES10)|RXMG_(?!ZZZX000)|ACG_' if 'ACG' in config.PREDICTORREGEX else CCSGMA
           }

assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'

PATIENT_ID=X.PATIENT_ID
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
    if undersampling:
        ID_highcost= y.loc[y[config.COLUMNS[0]]==1].PATIENT_ID 
        y_lowcost = y.loc[y[config.COLUMNS[0]]==0].sample(y[config.COLUMNS].sum().values[0])
        y = pd.concat([y_lowcost, y.loc[y[config.COLUMNS[0]]==1]])
        X = pd.concat([X.loc[X.PATIENT_ID.isin(y_lowcost.PATIENT_ID)] , X.loc[X.PATIENT_ID.isin(ID_highcost)]])
        predictors=X.columns
        df=pd.merge(X, y, on='PATIENT_ID')
        df=df.sample(frac=1,random_state=42).reset_index()
        X,y=df[predictors],df[config.COLUMNS]
    # y=y[config.COLUMNS]
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
    undersampling=False
    
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
    
except:
    pass


 
#%%

#%%
for key, val in variables.items():
    print('STARTING ',key, val)
    if not val:
        continue
    if Path(os.path.join(config.MODELPATH,key)).is_dir(): #the model is already there
        continue
    Xx=X.filter(regex=val, axis=1)
    
    if key=='DemoDiagPharmaBinary':
        continue
        print(Xx.PHARMA_Transplant.describe())
        Xx[[c for c in Xx if c.startswith('PHARMA')]]=(Xx[[c for c in Xx if c.startswith('PHARMA')]]>0).astype(int)
        print(Xx.PHARMA_Transplant.describe())
    
    X_train, X_test, y_train, y_test = train_test_split( Xx, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_test2, y_test, y_test2 = train_test_split( X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    
    
    print('Sample size ',len(y_train))
    print('---------------------------------------------------'*5)
 

    """ FIT MODEL """
    np.random.seed(seed_hparam)
    print('Seed ', seed_hparam)
    
    model_name=config.MODELPATH+key

    if tuner=='bayesian':
        print('Tuner: Bayesian')
        tuner= kt.BayesianOptimization(config.build_model,
                             objective=kt.Objective("val_loss", direction="min"),
                              max_trials=10,
                              overwrite=False,
                              num_initial_points=4,
                             directory=model_name+'_search',
                             project_name=key,   
                             
                             )
    else:
        print('Tuner: Random')
        tuner= kt.RandomSearch(config.build_model,
                             objective=kt.Objective("val_loss", direction="min"),
                              max_trials=10,
                              overwrite=False,
                             directory=model_name+'_search',
                             project_name=key,   
                             
                             )
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    tuner.search(X_train, y_train,epochs=30, validation_split=0.2,callbacks=[stop_early],
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
PATIENT_ID=X.PATIENT_ID
if hasattr(config, 'target_binarizer'):
    y=config.target_binarizer(y)
else:
    y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
   
y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y
#%%
table=pd.DataFrame()
for key, val in variables.items():
    Xx=X.copy()
    if not val:
        continue
    if key=='DemoDiagPharmaBinary':
        print(Xx.PHARMA_Transplant.describe())
        Xx[[c for c in Xx if c.startswith('PHARMA')]]=(Xx[[c for c in Xx if c.startswith('PHARMA')]]>0).astype(int)
        print(Xx.PHARMA_Transplant.describe())
    
    probs,_=predict(key,experiment_name=config.EXPERIMENT,year=2018,
                      X=Xx.filter(regex=val, axis=1), y=y)
    auc=roc_auc_score(probs.OBS,probs.PRED)
    recall, ppv, _, _ = performance(obs=probs.OBS, pred=probs.PRED, K=20000)
    brier=brier_score_loss(y_true=probs.OBS, y_prob=probs.PRED)
    ap=average_precision_score(probs.OBS,probs.PRED)
    table=pd.concat([table,
                     pd.DataFrame.from_dict({'Model':[key], 'AUC':[ auc], 'AP':[ap],
                      'R@20k': [recall], 'PPV@20K':[ppv], 
                      'Brier':[brier]})])
    
#%%
print(table.to_markdown(index=False))

