#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:18:33 2021

@author: aolza
"""
EXPERIMENT='sin_residenciado'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
INDICEPRIVACION=False
COLUMNS=['urgcms']
PREVIOUSHOSP=[]
EXCLUDE=['hdia','nbinj']
RESOURCEUSAGE=True
