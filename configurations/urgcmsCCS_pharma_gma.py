#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:55:10 2022

@author: aolza
"""

EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['urgcms']
TRACEBACK=True

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|AGE_[0-9]+$|FEMALE|CCS(?!2601|2602|2603|2604|2605|2606|2607|2608|2609|2610|2611|2612|2613|2614|2615|2618|2619|2620|2621)[0-9]+|CCSONCOLO'
INDICEPRIVACION=False
COLUMNS=['urgcms']#variable respuesta
EXCLUDE=['nbinj']
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
PHARMACY=True
""" GMA """
GMA=True
GMACATEGORIES=True
GMAOUTFILES={2016: ['gma/outGMA_2016_h.txt','gma/outGMA_2016_m.txt'],
             2017: ['gma/outGMA_2017_h.txt','gma/outGMA_2017_m.txt']}


""" CCS"""
CCS=True
ICDFILES={2016:'ccs/dx_in_2016.txt',
          2017:'ccs/dx_in_2017.txt'}
ICDTOCCSFILES={'ICD10CM':'ccs/translate_icd10cm_ccs_2018.csv',
               'ICD9':'ccs/translate_icd9_ccs_2015.csv'}
CCSFILES={2016:'newCCS2016.csv',
          2017: 'newCCS2017.csv'}

ATCFILES={2016:'pharma2016.csv',
                    2017: 'pharma2017.csv'}
