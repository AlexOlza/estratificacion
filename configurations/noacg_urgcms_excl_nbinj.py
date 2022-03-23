#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:37:21 2022

@author: aolza
"""

EXPERIMENT='noacg_urgcms_excl_nbinj'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|EDC_|FRAILTY|RXMG_'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
PREVIOUSHOSP=[]
EXCLUDE=['nbinj']
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
