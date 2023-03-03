#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:24:08 2022

@author: aolza
"""
EXPERIMENT=__name__.split('.')[-1]
CONFIGNAME=EXPERIMENT+'.py'
COLUMNS=['COSTE_TOTAL_ANO2']
TRACEBACK=True
EXCLUDE=[]
"""PREDICTORS"""
PREDICTORREGEX=r'PATIENT_ID|AGE_[0-9]+$|FEMALE|CCS(?!260[1-9]+|261[0-9]+|2620|2621|25[5-9]+)[0-9]+|CCSONCOLO'
INDICEPRIVACION=False
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
PHARMACY=True
""" GMA """
GMAOUTFILES={2016: ['gma/outGMA_2016_h.txt','gma/outGMA_2016_m.txt'],
             2017: ['gma/outGMA_2017_h.txt','gma/outGMA_2017_m.txt']}
GMA=True
GMACATEGORIES=True
GMA_ADDITIONAL_COLS=['GMA_peso-ip']
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

def categorize_ip(X):
    import numpy as np
    import pandas as pd
    print('Categorizing the GMA variable peso-ip')
    dff=X.copy()
    original_GMA_variables=[c for c in X if 'GMA' in c]
    quantiles=dff['GMA_peso-ip'].quantile([0.5,0.8,0.95,0.99])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.50],2,1)
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.80],3,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.95],4,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.99],5,dff['complejidad'])
    dff=pd.concat([dff, pd.get_dummies(dff.complejidad)],axis=1)
    dff.rename(columns={i:f'GMA_peso_quantile_{i}' for i in [1,2,3,4,5]},inplace=True)
    dff.drop(['complejidad'], axis=1,inplace=True)
    dff.drop(original_GMA_variables, axis=1,inplace=True)
    return dff

def modifyData(X,y):
    X=categorize_ip(X)
    return X, y