#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
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

from dataManipulation.dataPreparation import getData
from dataManipulation.pharmacy.ATC_CCS import generatePharmacyData
#%%

X,y=getData(year)
X=generatePharmacyData(year, X)



