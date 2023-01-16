#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:08:49 2023

@author: aolza
"""
from anchor import utils
from anchor import anchor_tabular

#%%
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcmsCCS_pharma'
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

#%%

import numpy as np
import alibi
from tensorflow import keras
import tensorflow as tf
alibi.explainers.__all__
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#%%
X,y=getData(2017)
feature_names=X.drop('PATIENT_ID',axis=1).columns
#%%
modelpath='/home/aolza/Desktop/estratificacion/models/urgcmsCCS_pharma/nested_neural'
model=keras.models.load_model(modelpath+'/DemoDiagPharmaBinary')
#%%
catnames={i:['0','1'] for i,v in enumerate(feature_names) if not 'CCS' in v}
explainer = anchor_tabular.AnchorTabularExplainer(['ing','sano'],
                                                  list(feature_names),
                                                  X.drop('PATIENT_ID',axis=1).to_numpy(),
                                                  catnames)
#%%
class_predictor= lambda x: [np.array([model.predict(x), 1-model.predict(x)]).ravel().reshape(1,-2).argmax()]
idx = 0
np.random.seed(1)
print('Prediction: ', explainer.class_names[class_predictor(instance)[0]])
exp = explainer.explain_instance(instance, class_predictor, threshold=0.95)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())

