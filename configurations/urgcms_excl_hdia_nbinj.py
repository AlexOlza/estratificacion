#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:32:19 2021

@author: aolza
"""
# import sys
# from configurations.default import *

EXPERIMENT='urgcms_excl_hdia_nbinj'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
PREDICTORS=True #FIXME may be obsolete
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
PREVIOUSHOSP=[]
EXCLUDE=['hdia','nbinj']
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
