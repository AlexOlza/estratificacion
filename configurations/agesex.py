#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:35:59 2022

@author: aolza
"""


EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['urg']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
EXCLUDE=[]
EXCLUDEOSI=[]
RESOURCEUSAGE=False

""" CCS"""
CCS=True
ICDFILES={2016:'ccs/dx_in_2016.txt',
          2017:'ccs/dx_in_2017.csv'}
ICDTOCCSFILES={'ICD10CM':'ccs/translate_icd10cm_ccs_2018.csv',
               'ICD9':'ccs/translate_icd9_ccs_2015.csv'}
CCSFILES={2016:'CCS2016.csv',
          2017: 'CCS2017.csv'}