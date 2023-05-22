#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:22:05 2022

@author: alex
"""
EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_'
INDICEPRIVACION=False
COLUMNS=['COSTE_TOTAL_ANO2']#variable respuesta
EXCLUDE=[]
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
