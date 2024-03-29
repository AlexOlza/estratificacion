#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:24:08 2022

@author: alex
"""
EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['COSTE_TOTAL_ANO2']
TRACEBACK=True
EXCLUDE=[]
"""PREDICTORS"""
PREDICTORREGEX=r'PATIENT_ID|AGE_[0-9]+$|FEMALE|CCS(?!2601|2602|2603|2604|2605|2606|2607|2608|2609|2610|2611|2612|2613|2614|2615|2618|2619|2620|2621)[0-9]+|CCSONCOLO'
INDICEPRIVACION=False
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False

""" CCS"""
CCS=True
ICDFILES={2016:'ccs/dx_in_2016.txt',
          2017:'ccs/dx_in_2017.txt'}
ICDTOCCSFILES={'ICD10CM':'ccs/translate_icd10cm_ccs_2018.csv',
               'ICD9':'ccs/translate_icd9_ccs_2015.csv'}
CCSFILES={2016:'newCCS2016.csv',
          2017: 'newCCS2017.csv'}