#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
import sys
try:
    model_name=sys.argv[1]
    year=int(sys.argv[2])
except:
    model_name=input('MODEL NAME (example: logistic20220118_132612): ')
    year=int(input('YEAR YOU WANT TO PREDICT:'))

sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.utility import configure
import joblib as job
configname='/home/aolza/Desktop/estratificacion/configurations/used/{0}.json'.format(model_name)
configuration=configure(configname,TRACEBACK=False, VERBOSE=True)
try:
    experiment_name=config.EXPERIMENT
except:
    experiment_name=input('EXPERIMENT NAME (example: urgcms_excl_nbinj): ')

    
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
def predict_save(yr,model,model_name,X,y,**kwargs):
    columns=kwargs.get('columns',config.COLUMNS[0])
    verbose=kwargs.get('verbose',config.VERBOSE)
    from more_itertools import sliced
    CHUNK_SIZE = 50000 #TODO experiment with this to try to speed up prediction
    
    index_slices = sliced(range(len(X)), CHUNK_SIZE)
    i=0
    n=len(X)/CHUNK_SIZE
    filename=config.PREDPATH+'/{0}__{1}.csv'.format(model_name,yr)
    with open(filename,'w') as predFile:
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
    print('saved',filename) 
    return filename
 
    #%%
import csv
import pandas as pd
from dataManipulation.dataPreparation import getData
from sklearn.metrics import roc_auc_score
import numpy as np

#%%
if __name__=='__main__':
        
    # FIXME STRUCT KEYS TO INT, FIX GENERARTABLASVARIABLES.RETRIEVE INDICE
    
    modelfilename='/home/aolza/Desktop/estratificacion/models/{1}/{0}.joblib'.format(model_name,experiment_name)

    model=job.load(modelfilename)

    Xx,Yy=getData(year-1)

    predFilename=predict_save(year, model,model_name, Xx, Yy, verbose=False)
    probs=pd.read_csv(predFilename)
    print(probs.head())

    print('auc ',roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)) 
