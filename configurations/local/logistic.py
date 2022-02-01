#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:56 2021

@author: aolza
"""
from configurations.default import *
import os

import importlib
import sys
try:
    chosen_config=sys.argv[1]
    experiment=sys.argv[2]
except:
    experiment=input('EXPERIMENT NAME (for example urgcms_excl_hdia_nbinj): ')
experiment='configurations.'+experiment

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

ALGORITHM='logistic'
CONFIGNAME='logistic.py'

MODELPATH=MODELSPATH+EXPERIMENT+'/'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)


