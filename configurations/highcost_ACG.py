#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 2022

@author: aolza
"""

EXPERIMENT='highcost_ACG'

"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_(?!NUR11|RES10)|HOSDOM|FRAILTY|RXMG_(?!ZZZX000)|INGRED_14GT'
INDICEPRIVACION=False
COLUMNS=['COSTE_TOTAL_ANO2']#variable respuesta
EXCLUDE=[]
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 
RESOURCEUSAGE=False
K=20000
def target_binarizer(y, K=K, column=COLUMNS):
    import pandas as pd
    import numpy as np
    cutoff_value=y[column].nlargest(K,columns=column).min()
    y[column[0]]=np.where(y[column]>=cutoff_value,1,0).ravel()
    return y
