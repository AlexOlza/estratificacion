#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:56 2021

@author: aolza
"""

EXPERIMENT='almeria'
CONFIGNAME='almeria.py'
COLUMNS=['urg']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
INDICEPRIVACION=False
COLUMNS=['urg']#variable respuesta
PREVIOUSHOSP=[]
EXCLUDE=[]
EXCLUDEOSI=[]
RESOURCEUSAGE=False

