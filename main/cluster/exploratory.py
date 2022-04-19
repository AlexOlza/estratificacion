#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:27:22 2022

@author: aolza
"""
import re
import pandas as pd

PROJECT='estratificacion'
ROOTPATH='/home/aolza/Desktop/estratificacion/'
DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
INDISPENSABLEDATAPATH=DATAPATH+'indispensable/'

f='ccs_dx_icd10cm_2017.csv'

df=pd.read_csv(INDISPENSABLEDATAPATH+'ccs/'+f, dtype=str)

for c in df:
    print(f'{c} has {len(df[c].unique())} unique values')

print('CCS CATEGORIES ARE: ')
print(df['CCS CATEGORY DESCRIPTION'].unique())

