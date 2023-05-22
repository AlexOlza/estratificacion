#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:09:10 2023

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

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
modelpath='/home/alex/Desktop/estratificacion/models/urgcmsCCS_pharma/nested_neural'
model=keras.models.load_model(modelpath+'/DemoDiagPharmaBinary')
#%%
predict_fn = lambda x: np.array([model.predict(x), 1-model.predict(x)]).ravel().reshape(1,-2)

x_train=np.random.randint(0,2,model.input_shape[1])
shape = (1,) + x_train.shape
cf = alibi.explainers.Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=0.5,
                                     # categorical_names=catnames,
                    target_class='other', max_iter=1000, early_stop=50, lam_init=1e-1,
                    max_lam_steps=10, tol=0.05, learning_rate_init=0.1,
                    # feature_range=(0, 1),
                    eps=0.01, init='identity',
                    decay=True, write_dir=None, debug=True)

pred=model.predict(X.iloc[0].drop('PATIENT_ID').values.reshape(1,-1))
explanation = cf.explain(instance) #ValueError: cannot reshape array of size 0 into shape (1,1,319)
pred=predict_fn(x_train.reshape(1,-1))
explanation = cf.explain(x_train.reshape(1,-1))
# #%%
# # f takes the data format your model uses and encodes it into the format alibi likes
# def f(X: np.ndarray, **kwargs) -> np.ndarray:  # use **kwargs for any other information needed to do the conversion
#     Z_num = extract_numeric(X, **kwargs)  # extract columns like 49.5,  Z_num is now a homogenous array of numbers
#     Z_cat = extract_cat(X, **kwargs)  # take columns like 'Male' and convert to 0, Z_cat is now a homogenous array of numbers
#     Z = combine(Z_num, Z_cat, **kwargs)  # concatenate columns in the right order
#     return Z

# # the f inverse function takes the encoded data alibi needs and decodes it into the format your model uses.
# def f_inv(Z: np.ndarray, **kwargs) -> np.ndarray:
#     ... # do similar operations as above
#     return Z
# def M_hat(Z: np.ndarray) -> np.ndarray: # Z here is the encoded data that doesn't work with your model
#     X = f_inv(Z) # X is now the data that does work with your model
#     pred = M(X)
#     return pred
#%%

from alibi.explainers import AnchorTabular
from alibi.utils import gen_category_map
predict_fn(np.zeros([1, len(list(feature_names))]))
category_map = gen_category_map(X.sample(frac=0.0001))
catnames={k:[0,1] for k in feature_names }
catnames={i:['0','1'] for i in range(len(feature_names)) }
explainer = AnchorTabular(predict_fn, list(feature_names),categorical_names=catnames,
                         )
ref=X.drop('PATIENT_ID',axis=1).sample(frac=0.001)
explainer.fit(ref.to_numpy(),disc_perc=None,verbose=3)

#%%
idx = 714821
class_names = ['ing','sano']
print('Prediction: ', class_names[explainer.predictor(ref.loc[idx,:].to_numpy().reshape(1, -1))[0]])
pred=model.predict(ref.loc[idx,:].to_numpy().reshape(1, -1))
instance=ref.to_numpy()[100].reshape(1, -1)
explanation = explainer.explain(instance) #IndexError: boolean index did not match indexed array along dimension 0; dimension is 100 but corresponding boolean dimension is 1
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)