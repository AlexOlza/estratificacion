#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:56 2021

@author: aolza
"""
from configurations.default import *
import os
EXPERIMENT='urg'
MODELPATH+=EXPERIMENT+'/'
ALGORITHM='logistic'
CONFIGNAME='logUrgHdia.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
COLUMNS=['urg']
TRACEBACK=True
EXCLUDE='hdia'