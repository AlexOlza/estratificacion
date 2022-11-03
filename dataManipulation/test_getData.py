#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts tests wether getData can recover the data matrix we had before (with OLDBASE)
Created on Wed Jan 12 11:19:22 2022

@author: aolza
"""
import importlib
import sys
chosen_config='configurations.cluster.'+sys.argv[1]
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 


from dataManipulation.dataPreparation import getData
import pandas as pd
X,y=getData(2016, columns='DEATH_1YEAR')
# outcome=['urg']
# # outcome.remove('PATIENT_ID')

# _,yOLD=getData(2016,oldbase=True)

# df=pd.DataFrame()
# df['id']=y.PATIENT_ID
# df['OLD']=yOLD.ingresoUrg
# df['NEW']=y[outcome]

# print('Patients with different outcome with the new hospitalizatiton data:')
# print(df[df.OLD!=df.NEW])

# print('NOTE: SOME SMALL DIFFERENCES ARE NORMAL')
