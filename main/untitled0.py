#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:28:38 2021

@author: aolza
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:19:44 2021

@author: aolza
"""
import numpy as np
# rng = np.random.RandomState(0)
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
chosen_config='configurations.cluster.'+sys.argv[1]
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
#%%
""" BEGGINNING """
from dataManipulation.dataPreparation import getData
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
# np.random.seed(config.SEED)

pred16,y17=getData(2016,predictors=True)
assert len(config.COLUMNS)==1, 'This model is built for a single response variable! Modify config.COLUMNS'


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
try:
    X.drop('PATIENT_ID',axis=1,inplace=True)
except:
    pass

y=pd.concat([y17.loc[random_indices],y17.loc[ing_indices]])
y.drop(na_indices,axis=0,inplace=True)
y=np.where(y[config.COLUMNS]>=1,1,0)
print('Dropping NA')
print('Sample size ',len(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.9,random_state=config.SEED)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 100, num = 5)]
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
rg = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
if not isinstance(config.FOREST,RandomForestClassifier):
    print('exec happening')
    exec("estimator="+config.FOREST)
else:
    print('exec not needed')
    estimator=config.FOREST
forest = RandomizedSearchCV(estimator = estimator, 
                               param_distributions = rg,
                               n_iter = 3,
                               cv = config.CV, 
                               verbose=3,
                               random_state=config.SEED,
                               n_jobs =-1)
forest.fit(X_train,y_train)
#%%
""" SAVE TRAINED MODEL """
util.savemodel(config,forest.best_estimator_)
#%%
# import joblib
# reload=joblib.load('/home/aolza/Desktop/estratificacion/models/urgcms_excl_hdia_nbinj/randomForest20211215_163215.joblib')
# #
