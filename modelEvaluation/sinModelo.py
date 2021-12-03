#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:35:11 2021

@author: aolza
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
def recallppv(ytrue,ypred):
    c=confusion_matrix(y_true=ytrue, y_pred=ypred)
    recall=c[1][1]*100/(c[1][0]+c[1][1])
    ppv=c[1][1]*100/(c[0][1]+c[1][1])
    return(recall,ppv)
previous=[16]#16,,18
years=[17]#17,,19
N=[22405,100000]
for prev,yr in zip(previous,years):
    ingresosfile = "estratificacionDatos/ing20{0}-20{1}Activos.csv".format(prev,yr)
    datos = pd.DataFrame()
    for chunk in pd.read_csv(ingresosfile, chunksize=100000,
                             columns=['COSTE_TOTAL_ANO2','PATIENT_ID','ingresoUrg']):
        d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
        d['PATIENT_ID']=np.int64
        chunk= chunk.astype(d)
        # print(chunk.info())
        datos = pd.concat([datos, chunk], ignore_index=True)
        del chunk
    datos['algunIngresoUrg']=False
    datos.iloc[datos.loc[datos['ingresoUrg']>0].index, datos.columns.get_loc('algunIngresoUrg')]=True

    futurofile = "estratificacionDatos/ing20{0}-20{1}Activos.csv".format(prev+1,yr+1)
    futuro = pd.DataFrame()
    for chunk in pd.read_csv(futurofile,'df', chunksize=100000,
                             columns=['COSTE_TOTAL_ANO2','PATIENT_ID','ingresoUrg']):
        d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
        d['PATIENT_ID']=np.int64
        chunk= chunk.astype(d)
        # print(chunk.info())
        futuro = pd.concat([futuro, chunk], ignore_index=True)
        del chunk
    futuro['algunIngresoUrgFut']=False
    futuro.iloc[futuro.loc[futuro['ingresoUrg']>0].index, futuro.columns.get_loc('algunIngresoUrgFut')]=True
    ppv,ppving,rec,recing=[],[],[],[]
    for n in N:
        datos=datos.sort_values('COSTE_TOTAL_ANO2',ascending=False)
        futuro=futuro.sort_values('COSTE_TOTAL_ANO2',ascending=False)
        datos.reset_index(drop=True,inplace=True)
        futuro.reset_index(drop=True,inplace=True)
      
        datos['top1']=False
        datos.iloc[:n, datos.columns.get_loc('top1')]=True
        futuro['top1fut']=False
        futuro.iloc[:n, futuro.columns.get_loc('top1fut')]=True
        
        df=pd.merge(left=datos, right=futuro, left_on='PATIENT_ID', right_on='PATIENT_ID')   
        r,p=recallppv(df['top1fut'],df['top1'])
        ppv.append(p)
        rec.append(r)
    
        ring,ping=recallppv(df['algunIngresoUrgFut'],df['top1'])
        ppving.append(ping)
        recing.append(ring)

    print('Recall{0}Ac='.format(yr),rec)
    print('PPV{0}Ac='.format(yr),ppv)
    print('RecallIngresos{0}Ac='.format(yr),recing)
    print('PPVIngresos{0}Ac='.format(yr),ppving)
