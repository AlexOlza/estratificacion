#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
from python_settings import settings as config
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
def configure(configname=None):
    if not configname:
        configname=input('Enter saved configuration file path: ')
    with open(configname) as c:
        configuration=json.load(c)
    for k,v in configuration.items():
        if isinstance(v,dict):#for example ACGFILES
             configuration[k]= {int(yr):filename for yr,filename in v.items()}
    print(configuration)
    conf=Struct(**configuration)
    conf.TRACEBACK=True
    conf.VERBOSE=True
    if not config.configured:
        config.configure(conf) # configure() receives a python module
    assert config.configured
    return configuration
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
def predict_save(yr,model,X,y,columns=config.COLUMNS[0]):

    from more_itertools import sliced
    CHUNK_SIZE = 50000
    
    index_slices = sliced(range(len(X)), CHUNK_SIZE)
    i=0
    n=len(X)/CHUNK_SIZE
    with open(config.PREDFILES[yr-1],'w') as predFile:
        csv_out=csv.writer(predFile)
        csv_out.writerow(['PATIENT_ID','PRED','OBS'])
        for index_slice in index_slices:
            i+=1
            util.vprint(i,'/',n)
            chunk = X.iloc[index_slice] # your dataframe chunk ready for use
            ychunk=y.iloc[index_slice]
            predictions=model.predict_proba(chunk.drop('PATIENT_ID',axis=1))[:,1] #Probab of being top1
            for element in zip(chunk['PATIENT_ID'],ychunk['PATIENT_ID'],predictions,ychunk[columns]):
                csv_out.writerow(element)
                del element
    print('saved') 
 
    #%%
import json
import joblib as job
import csv
import pandas as pd
if not config.configured:
    configure()
import configurations.utility as util
# from dataManipulation.generarTablasVariables import load
from dataManipulation.dataPreparation import getData
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import linear_model
from main import SafeLogisticRegression

#%%
if __name__=='__main__':
        
    # FIXME STRUCT KEYS TO INT, FIX GENERARTABLASVARIABLES.RETRIEVE INDICE
    configname='configurations/local/used/logistic20211207_132452.json'
    configuration=configure(configname)
    modelfilename='/home/aolza/Desktop/estratificacion/models/OLDBASE/logistic20211207_132452.joblib'
    model=job.load(modelfilename)
    
    Xx,Yy=getData(2017,oldbase=True)
    cols=list(Xx.columns)
    if 'ingresoUrg' in cols: cols.remove('ingresoUrg')
  
    predict_save(2018, model, Xx, Yy,columns='ingresoUrg')
    probs=pd.read_csv(config.PREDFILES[2017])

    auc=roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)
    oldprobs=pd.read_csv('untracked/predecirIngresos/logistica/urgSinPrevio18.csv')
    oldauc=roc_auc_score(np.where(Yy.ingresoUrg>=1,1,0), oldprobs.PROB_ING)

