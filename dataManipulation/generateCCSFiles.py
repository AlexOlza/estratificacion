#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:07:04 2022

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
year=int(sys.argv[3])

import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData, generateCCSData
#%%

X,y=getData(year,  predictors='PATIENT_ID|AGE_|FEMALE')
assert 'ONCOLO' in X
# _ , _ =generateCCSData(year,  X)

