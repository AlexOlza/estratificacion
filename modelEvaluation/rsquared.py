#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:34:34 2021

@author: aolza
"""
def recallppv(ytrue,ypred):
    c=confusion_matrix(y_true=ytrue, y_pred=ypred)
    recall=c[1][1]*100/(c[1][0]+c[1][1])
    ppv=c[1][1]*100/(c[0][1]+c[1][1])
    return(recall,ppv)
from sklearn.metrics import r2_score,explained_variance_score,mean_squared_error
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
previous=[17]#16,,18
years=[18]#17,,19
for prev,yr in zip(previous,years):
    # if previous!=17:
    #     continue
    ingresosfile = "estratificacionDatos/ing20{0}-20{1}Activos.csv".format(prev,yr)
    filename = "estratificacionDatos/20{0}-20{1}Activos.csv".format(prev,yr)
    hgbfile="linealPython/predictions/hgb20{0}AgeSexDiagRxActivos.csv".format(yr)
    lassofile="linealPython/predictions/lasso20{0}AgeSexDiagRxActivos.csv".format(yr)
    linfile="linealPython/predictions/pred20{0}AgeSexDiagRxActivos.csv".format(yr) #lineal
    # adalinfile="../cluster/adalin/predictions/adalinsq20{0}AgeSexDiagRxActivos.csv".format(yr)
    rffile="../cluster/coste/rfund/predictions/rfund3020{0}AgeSexDiagRxActivos.csv".format(yr)
    tweediefile="tweedie/predictions/tweedie1.2PredAgeSexDiagRx20{0}.csv".format(yr)
    gammafile="gamma/predictions/gamma20{0}AgeSexDiagRxActivos.csv".format(yr)
    # nnfile="neuralCoste/inlenpredhid155_vallos/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155hid155_vallos_restore_seed42/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid775hid775_vallos_restore_seed42/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_patience5/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_100epoch_lr001_dec0001/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec00005/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr0001_nesterov/pred20{0}AgeSexDiagRxActivos.csv".format(yr)# nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec0/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_nesterov/pred20{0}AgeSexDiagRxActivos.csv".format(yr)# nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec0/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr005_nesterov/pred20{0}AgeSexDiagRxActivos.csv".format(yr)# nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec0/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_nadam_batch256_nonorm/pred20{0}AgeSexDiagRxActivos.csv".format(yr)#
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_nadam_batch512_nonorm/pred20{0}AgeSexDiagRxActivos.csv".format(yr)#
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec00005_batch256/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inlenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_nadam_batch256_lr001_nonorm/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/inhalflenpredhid155_vallos_restore_seed42_lin_allheseeded_elu_800epoch_lr001_dec00005_valsplit01/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/talosfinde/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    nnfile="neuralCoste/batchnormAltencoding/pred20{0}AgeSexDiagRxActivos.csv".format(yr)
    # filenames=[hgbfile,lassofile,linfile,rffile,tweediefile,gammafile,nnfile]
    filenames=[linfile,nnfile]
    #%%  
    datos = pd.DataFrame()
    for chunk in pd.read_csv(filename, chunksize=100000):
        d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
        d['PATIENT_ID']=np.int64
        chunk= chunk.astype(d)
        datos = pd.concat([datos, chunk], ignore_index=True)

    datos['algunIngresoUrg']=False
    datos.iloc[datos.loc[datos['ingresoUrg']>0].index, datos.columns.get_loc('algunIngresoUrg')]=True

    for predfile in filenames:
        pred = pd.DataFrame()
        for chunk in pd.read_csv(predfile,chunksize=100000):
            pred = pd.concat([pred, chunk], ignore_index=True)
            del chunk
        
        #%%
        datos=datos.sort_values('PATIENT_ID',ascending=False)
        datos.reset_index(drop=True,inplace=True)
        pred=pred.sort_values('PATIENT_ID',ascending=False)
        pred.reset_index(drop=True,inplace=True)
        print(predfile)
        print('R2=',r2_score(datos.COSTE_TOTAL_ANO2, pred.COSTE))
        print('expl var=',explained_variance_score(datos.COSTE_TOTAL_ANO2, pred.COSTE))
        print('RMSE=',np.sqrt(mean_squared_error(datos.COSTE_TOTAL_ANO2, pred.COSTE)))
        lin=LinearRegression()
        lin.fit(pred.COSTE.to_numpy().reshape(-1, 1),datos.COSTE_TOTAL_ANO2 )
        print('R2 linear regression ',lin.score(pred.COSTE.to_numpy().reshape(-1, 1),datos.COSTE_TOTAL_ANO2 ))
        
        
        datos=datos.sort_values('COSTE_TOTAL_ANO2',ascending=False)
        datos.reset_index(drop=True,inplace=True)
        N=[22405,44810,50000,67215,89620,int(1e5),112025]
        datos['top1']=False
        datos.iloc[:N[0], datos.columns.get_loc('top1')]=True
        
        ppv,ppving,rec,recing=[],[],[],[]
        
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
        
            ring,ping=recallppv(df['algunIngresoUrg'],df['top1pred'])
            ppving.append(ping)
            recing.append(ring)

        print('Recall{0}Ac='.format(yr),rec)
        print('PPV{0}Ac='.format(yr),ppv)
        print('RecallIngresos{0}Ac='.format(yr),recing)
        print('PPVIngresos{0}Ac='.format(yr),ppving)