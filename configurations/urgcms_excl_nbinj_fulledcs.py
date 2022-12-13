#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 2022

@author: aolza
"""

EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
FULLEDCS=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
EXCLUDE=['nbinj']
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
