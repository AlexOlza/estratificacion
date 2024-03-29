#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:56 2021

@author: alex
"""
from configurations.default import *#contains paths and verbosity
import importlib
import os
import sys
try:
    chosen_config=sys.argv[1]
    experiment='configurations.'+sys.argv[2]
except:
    chosen_config=input('CONFIG FILENAME: ')#example logisticOLDBASE
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

MODELPATH=MODELSPATH+EXPERIMENT+'/'
ALGORITHM='logistic'
CONFIGNAME='logisticOLDBASE.py'#FIXME may be obsolete, or it should be 
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
