#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:05:14 2021

@author: aolza
"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
def performance(ytrue,probs,key='top1',**kwargs):
    N=kwargs.get('N',0)
    auc=roc_auc_score(ytrue, probs)
    print('AUC=',auc)
    # fpr, tpr, thresholds = roc_curve(ytrue, probs)
    # print('TPR-FPR=',max(tpr-fpr))#youden index 
    # topt=thresholds[np.argmax( tpr-fpr)]
    # print('Optimal threshold=',topt)
    #%%
    print('Confusion Matrix')
    if N>0:
        #El punto de corte es la N-ésima mayor probabilidad
        orderedProbs=sorted(probs,reverse=True)
        cutoff=orderedProbs[N]
        print('Cutoff probability ({0} values): {1}'.format(N, cutoff))
        newclasses=probs>=cutoff
    print('longitud listado ',sum(newclasses))   
    c=confusion_matrix(y_true=ytrue, y_pred=newclasses)
    print(c)
    
    print('Percentage Confusion Matrix')
    print('          False     True')
    print('False   {0}     {1}'.format(round(c[0][0]*100/(c[0][0]+c[0][1]),3), 
                                       round(c[0][1]*100/(c[0][0]+c[0][1]),3)))
    print('True    {0}     {1}'.format(round(c[1][0]*100/(c[1][0]+c[1][1]),3), 
                                       round(c[1][1]*100/(c[1][0]+c[1][1]),3)))
    print('\n'*3)
    recall=c[1][1]*100/(c[1][0]+c[1][1])
    ppv=c[1][1]*100/(c[0][1]+c[1][1])
    return(recall,ppv)
def recallppv(ytrue,ypred):
    c=confusion_matrix(y_true=ytrue, y_pred=ypred)
    recall=c[1][1]*100/(c[1][0]+c[1][1])
    ppv=c[1][1]*100/(c[0][1]+c[1][1])
    return(recall,ppv)
#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
previous=[17]#16,,18
years=[18]#17,,19
rec,ppv=[],[]
for prev,yr in zip(previous,years):
    # if previous!=17:
    #     continue
    filename = "estratificacionDatos/ing20{0}-20{1}Activos.csv".format(prev,yr)
    hgbfile="clasificacion/hgbsmallparamspace/probtop1_20{0}.csv".format(yr)
    rffile="clasificacion/randomForest/predictions/probtop1_20{0}recall.csv".format(yr)
    logfile="logistica/logTop1PredAgeSexDiagRx20{0}Activos.csv".format(yr)
    nbfile="clasificacion/naiveBayes/nbTop1PredAgeSexDiagRx20{0}Activos.csv".format(yr)
    files=[hgbfile,rffile,logfile,nbfile]
    N=[22405,44810,50000,67215,89620,int(1e5),112025]
    datos = pd.DataFrame()
    for chunk in pd.read_csv(filename, chunksize=100000):
        d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
        d['PATIENT_ID']=np.int64
        chunk= chunk.astype(d)
        datos = pd.concat([datos, chunk], ignore_index=True)

    datos=datos.sort_values('COSTE_TOTAL_ANO2',ascending=False)
    datos.reset_index(drop=True,inplace=True)
    datos['top1']=False
    datos.iloc[:N[0], datos.columns.get_loc('top1')]=True
    datos['algunIngresoUrg']=False
    datos.iloc[datos.loc[datos['ingresoUrg']>0].index, datos.columns.get_loc('algunIngresoUrg')]=True
    
    for predfile in files:
        ppv,ppving,rec,recing=[],[],[],[]
        
        #%%  
        print(predfile)    
        pred = pd.DataFrame()
        for chunk in pd.read_csv(predfile,chunksize=100000):
            pred = pd.concat([pred, chunk], ignore_index=True)
            del chunk
        
        datos=datos.sort_values('PATIENT_ID',ascending=False)
        datos.reset_index(drop=True,inplace=True)
        pred=pred.sort_values('PATIENT_ID',ascending=False)
        pred.reset_index(drop=True,inplace=True)
        for n in N:
            if n!=N[0]:
                continue
            
            df=pd.merge(left=datos, right=pred, left_on='PATIENT_ID', right_on='PATIENT_ID')
            r,p=performance(df['top1'],df['PROB_TOP1'],N=n)
            ppv.append(p)
            rec.append(r)
            orderedProbs=sorted(df['PROB_TOP1'],reverse=True)
            cutoff=orderedProbs[n]
            df['top1pred']=df['PROB_TOP1']>=cutoff
            ring,ping=recallppv(df['algunIngresoUrg'],df['top1pred'])
            ppving.append(ping)
            recing.append(ring)
        print('Recall{0}Ac='.format(yr),rec)
        print('PPV{0}Ac='.format(yr),ppv)
        print('RecallIngresos{0}Ac='.format(yr),recing)
        print('PPVIngresos{0}Ac='.format(yr),ppving)
    
