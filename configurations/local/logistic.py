#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:56 2021

@author: aolza
"""
from configurations.default import *
import os
EXPERIMENT='urgcms_excl_hdia_nbinj'
MODELPATH+=EXPERIMENT+'/'
ALGORITHM='logistic'
CONFIGNAME='logistic.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
PREDFILES={yr: os.path.join(PREDPATH,'{1}{0}.csv'.format(yr,ALGORITHM)) for yr in [2016,2017,2018]}
COLUMNS=['urg']
TRACEBACK=True
EXCLUDE=None