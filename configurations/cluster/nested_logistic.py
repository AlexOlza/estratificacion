#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:55:17 2022

@author: alex
"""
from configurations.default import *
import os

import importlib
import sys
try:
    chosen_config=sys.argv[1]
    experiment='configurations.'+sys.argv[2]
except:
    experiment=input('EXPERIMENT NAME: ')#example urgcms_excl_hdia_nbinj

"""THIS EMULATES 'from experiment import *' USING IMPORTLIB 
info: 
    https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
"""
mdl=importlib.import_module(experiment,package='estratificacion')
# is there an __all__?  if so respect it
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    # otherwise we import all names that don't begin with _
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
globals().update({k: getattr(mdl, k) for k in names}) #brings everthing into namespace
# print(names)

ALGORITHM='nested_logistic'
CONFIGNAME='nested_logistic.py'
USEDCONFIGPATH+=EXPERIMENT+'/'
MODELPATH=MODELSPATH+EXPERIMENT+'/'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
PREDFILES={yr: os.path.join(PREDPATH,'{1}{0}.csv'.format(yr,ALGORITHM)) for yr in [2016,2017,2018]}
FIGUREPATH=os.path.join(ROOTPATH,'figures',EXPERIMENT)
METRICSPATH=os.path.join(METRICSPATH,EXPERIMENT)


