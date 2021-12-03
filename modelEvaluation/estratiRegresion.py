#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914
MODIFICADO PARA CREAR LA TABLA DEL RESUMEN EN OVERLEAF
Created on Thu Jun 17 11:03:02 2021

@author: aolza
"""
#%%
def recallppv(ytrue,ypred):
    c=confusion_matrix(y_true=ytrue, y_pred=ypred)
    recall=c[1][1]*100/(c[1][0]+c[1][1])
    ppv=c[1][1]*100/(c[0][1]+c[1][1])
    return(recall,ppv)
from sklearn.metrics import r2_score,explained_variance_score
#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
previous=[17]#16,,18
years=[18]#17,,19
for prev,yr in zip(previous,years):
    # if previous!=17:
    #     continue
    filename = "estratificacionDatos/20{0}-20{1}Activos.csv".format(prev,yr)
    # predfile="linealPython/predictions/hgb20{0}AgeSexDiagRxActivos.csv".format(yr)
    # predfile="linealPython/predictions/lasso20{0}AgeSexDiagRxActivos.csv".format(yr)
    predfile="linealPython/predictions/pred20{0}AgeSexDiagRxActivos.csv".format(yr) #lineal
    # predfile="../cluster/adalin/predictions/adalinsq20{0}AgeSexDiagRxActivos.csv".format(yr)
    # predfile="../cluster/coste/rfund/predictions/rfund3020{0}AgeSexDiagRxActivos.csv".format(yr)
    # predfile="tweedie/predictions/tweedie1.2PredAgeSexDiagRx20{0}.csv".format(yr)
    # predfile="gamma/predictions/gamma20{0}AgeSexDiagRxActivos.csv".format(yr)
    #%%  
    
    datos = pd.DataFrame()
    for chunk in pd.read_csv(filename, chunksize=100000,columns=['COSTE_TOTAL_ANO2','PATIENT_ID']):
        datos = pd.concat([datos, chunk], ignore_index=True)
        del chunk
    
    pred = pd.DataFrame()
    for chunk in pd.read_csv(predfile,chunksize=100000):
        pred = pd.concat([pred, chunk], ignore_index=True)
        del chunk
    
    #%%
    datos=datos.sort_values('PATIENT_ID',ascending=False)
    datos.reset_index(drop=True,inplace=True)
    pred=pred.sort_values('PATIENT_ID',ascending=False)
    pred.reset_index(drop=True,inplace=True)
    print('R2=',r2_score(datos.COSTE_TOTAL_ANO2, pred.COSTE))
    print('expl var=',explained_variance_score(datos.COSTE_TOTAL_ANO2, pred.COSTE))
    datos=datos.sort_values('COSTE_TOTAL_ANO2',ascending=False)
    datos.reset_index(drop=True,inplace=True)
    N=[22405,44810,50000,67215,89620,int(1e5),112025]
    datos['top1']=False
    datos.iloc[:N[0], datos.columns.get_loc('top1')]=True
    
    ppv,rec=[],[]
    
    try:
        pred=pred.sort_values('PREDICCION_COSTE',ascending=False)
    #Esta excepción subsana el error de haber guardado ciertas predicciones de coste
    #con la etiqueta equivocada "CLASE"
    except KeyError:
        pred=pred.sort_values('COSTE',ascending=False)
    pred.reset_index(drop=True,inplace=True)
    for n in N:
        if n!=N[0]:
            continue
        pred['top1pred']=False
        pred.iloc[:n, pred.columns.get_loc('top1pred')]=True
        df=pd.merge(left=datos, right=pred, left_on='PATIENT_ID', right_on='PATIENT_ID')
        r,p=recallppv(df['top1'],df['top1pred'])
        ppv.append(p)
        rec.append(r)
    print('Recall{0}Ac='.format(yr),rec)
    print('PPV{0}Ac='.format(yr),ppv)
    
    
