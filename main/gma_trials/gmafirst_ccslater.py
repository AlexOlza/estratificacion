#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

@author: aolza
"""

import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from dataManipulation.dataPreparation import reverse_one_hot
from sklearn.preprocessing import OrdinalEncoder
X,y=getData(2016,
             CCS=True,
             PHARMACY=True,
             BINARIZE_CCS=True,
             GMA=True,
             GMACATEGORIES=True,
             GMA_DROP_DIGITS=0,
             additional_columns=[])

# X, y=X.sample(1000,random_state=1), y.sample(1000, random_state=1)
y=y[config.COLUMNS]
X=reverse_one_hot(X, integers=False)


enc = OrdinalEncoder(categories=[sorted(X.AGE.unique()),sorted(X.GMA.unique())],
                     dtype=np.int8)

X[['AGE','GMA']]=enc.fit_transform(X[['AGE','GMA']])

print('Sample size ',len(X))


estimator=LinearRegression(n_jobs=-1)
selector = RFE(estimator, n_features_to_select=50, step=3,verbose=1)
selector = selector.fit(X, y)
# selector.support_
# selector.ranking_
selected_features=X.columns[selector.support_]
print(selected_features)