#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts tests wether getData can recover the data matrix we had before (with OLDBASE)
Created on Wed Jan 12 11:19:22 2022

@author: aolza
"""
from dataManipulation.dataPreparation import getData
import pandas as pd
_,y=getData(2016,oldbase=False)
outcome=list(y.columns)
outcome.remove('PATIENT_ID')

_,yOLD=getData(2016,oldbase=True)

df=pd.DataFrame()
df['id']=y.PATIENT_ID
df['OLD']=yOLD.ingresoUrg
df['NEW']=y[outcome]

print('Patients with different outcome with the new hospitalizatiton data:')
print(df[df.OLD!=df.NEW])

print('NOTE: SOME SMALL DIFFERENCES ARE NORMAL')
