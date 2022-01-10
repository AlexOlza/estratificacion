#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.utility import configure
import joblib as job
configname='/home/aolza/Desktop/estratificacion/configurations/used/randomForest20211231_172918.json'
configuration=configure(configname,TRACEBACK=False, VERBOSE=True)

def performance(logistic,dat,predictors,probs=None,key='algunIngresoProg',file=None,mode='a',header='',AUC=True,**kwargs):
    global config
    if not config.configured:
        configuration,config=configure()
    N=kwargs.get('N',0)
    
    if file is not None:
        file = open(file, mode)
    if probs is None:
        probs=logistic.predict_proba(dat[predictors])[:,1] #Probab de ingreso
    print('N=',N,file=file)
    print('Numb unique probs ',len(np.unique(probs)),file=file)
    if AUC:
        auc=roc_auc_score(dat[key], probs)
        print('AUC=',auc,file=file)
    return(probs)
def predict_save(yr,model,X,y,**kwargs):
    columns=kwargs.get('columns',config.COLUMNS[0])
    verbose=kwargs.get('verbose',config.VERBOSE)
    from more_itertools import sliced
    CHUNK_SIZE = 50000 #TODO experiment with this to try to speed up prediction
    
    index_slices = sliced(range(len(X)), CHUNK_SIZE)
    i=0
    n=len(X)/CHUNK_SIZE
    with open(config.PREDFILES[yr-1],'w') as predFile:
        csv_out=csv.writer(predFile)
        csv_out.writerow(['PATIENT_ID','PRED','OBS'])
        for index_slice in index_slices:
            if verbose:
                i+=1
                print(i,'/',n)
            chunk = X.iloc[index_slice] # your dataframe chunk ready for use
            ychunk=y.iloc[index_slice]
            predictions=model.predict_proba(chunk.drop('PATIENT_ID',axis=1))[:,1] #Probab of being top1
            for element in zip(chunk['PATIENT_ID'],ychunk['PATIENT_ID'],predictions,ychunk[columns]):
                csv_out.writerow(element)
                del element
    print('saved',config.PREDFILES[yr-1]) 
 
    #%%
import csv
import pandas as pd
from dataManipulation.dataPreparation import getData
from sklearn.metrics import roc_auc_score
import numpy as np

#%%
if __name__=='__main__':
        
    # FIXME STRUCT KEYS TO INT, FIX GENERARTABLASVARIABLES.RETRIEVE INDICE
    
    modelfilename='/home/aolza/Desktop/estratificacion/models/urgcms_excl_hdia_nbinj/randomForest20211231_172918.joblib'

    model=job.load(modelfilename)

    Xx,Yy=getData(2017)

    predict_save(2018, model, Xx, Yy, verbose=False)
    probs=pd.read_csv(config.PREDFILES[2017])
    print(probs.head())

    print('auc ',roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)) 
    # oldprobs=pd.read_csv('untracked/predecirIngresos/logistica/urgSinPrevio18.csv')
    # oldauc=roc_auc_score(np.where(y.ingresoUrg>=1,1,0), oldprobs.PROB_ING)

    # print(model.score(Xx.drop('PATIENT_ID',axis=1),Yy.drop('PATIENT_ID',axis=1))) 
