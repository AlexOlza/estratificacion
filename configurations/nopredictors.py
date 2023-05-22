#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:41:27 2022

@author: alex
"""

EXPERIMENT='nopredictors'
CONFIGNAME='nopredictors.py'
COLUMNS=['urg']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
EXCLUDE=[]
EXCLUDEOSI=[]
RESOURCEUSAGE=False

""" CCS"""
CCS=True
ICDFILES={2016:'ccs/dx_in_2016.txt',
          2017:'ccs/dx_in_2017.txt'}
ICDTOCCSFILES={'ICD10CM':'ccs/translate_icd10cm_ccs_2018.csv',
               'ICD9':'ccs/translate_icd9_ccs_2015.csv'}
CCSFILES={2016:'CCS2016.csv',
          2017: 'CCS2017.csv'}