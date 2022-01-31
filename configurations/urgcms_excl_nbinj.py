#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 2022

@author: aolza
"""

EXPERIMENT='urgcms_excl_nbinj'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
PREDICTORS=True #FIXME may be obsolete
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
PREVIOUSHOSP=[]
EXCLUDE=['nbinj']
RESOURCEUSAGE=False
