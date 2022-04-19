#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:35:59 2022

@author: aolza
"""


EXPERIMENT='agesex'
CONFIGNAME='agesex.py'
COLUMNS=['urg']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
PREVIOUSHOSP=[]
EXCLUDE=[]
EXCLUDEOSI=[]
RESOURCEUSAGE=False
CCS=True