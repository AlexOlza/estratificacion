#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 26 Jan 2023

@author: aolza
"""

EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['COSTE_TOTAL_ANO2']
TRACEBACK=False
EXCLUDE=[]
"""PREDICTORS"""
PREDICTORREGEX=r'PATIENT_ID|AGE_[0-9]+$|FEMALE|CCS(?!260[1-9]+|261[0-9]+|2620|2621|25[5-9]+)[0-9]+|CCSONCOLO'
INDICEPRIVACION=False
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

def exclusion_criteria(X, y):
    import pandas as pd
    """ We exclude patients that have genitourinary conditions, and drop those columns """
    genitoCCS=X.filter(regex='PATIENT_ID|CCS(2[4-9]$|3[0-1]$|46$|16[3-9]$|17[0-9]$|18[0-9]$|19[0-6]$|215$)',axis=1)
    patients_to_exclude=X.loc[genitoCCS.filter(regex='CCS').sum(axis=1)>=1]
    percentwomen=100*patients_to_exclude.FEMALE.sum()/len(patients_to_exclude)
    print(f'We exclude {len(patients_to_exclude)} patients with genitourinary conditions, {percentwomen} % women')
    X=X.loc[~X.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)].drop(genitoCCS.filter(regex='CCS').columns,axis=1)
    y=y.loc[~y.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)]
    return (X,y)