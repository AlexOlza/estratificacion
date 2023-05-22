#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:32:34 2022

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

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
from sklearn.linear_model import LogisticRegression

from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
assert not 'AGE_85GT' in X.columns
#%%
from time import time
t0=time()
psm = PsmPy(X.sample(1000), treatment='FEMALE', indx='PATIENT_ID', exclude = [])

psm.logistic_ps(balance = True)
print('fitting ',time()-t0)
#%%
t0=time()
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.25)
print('matching ',time()-t0)
#%%
psm.plot_match(Title='Side by side matched controls', Ylabel='Number ofpatients', Xlabel= 'Propensity logit',names = ['treatment', 'control'],save=True)
#%%
