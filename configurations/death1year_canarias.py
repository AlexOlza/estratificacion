#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:55:10 2022

@author: aolza
"""

EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['DEATH_1YEAR']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|AGE_[0-9]+$|FEMALE|CCS(?!260[1-9]+|261[0-9]+|2620|2621|25[5-9]+)[0-9]+|CCSONCOLO'
INDICEPRIVACION=False
EXCLUDE=[]
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
PHARMACY=True


""" CCS"""
CCS=True
BINARIZE_CCS=True
ICDFILES={2016:'ccs/dx_in_2016.txt',
          2017:'ccs/dx_in_2017.txt'}
ICDTOCCSFILES={'ICD10CM':'ccs/translate_icd10cm_ccs_2018.csv',
               'ICD9':'ccs/translate_icd9_ccs_2015.csv'}
CCSFILES={2016:'newCCS2016.csv',
          2017: 'newCCS2017.csv'}

ATCFILES={2016:'pharma2016.csv',
                    2017: 'pharma2017.csv'}