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
def predict_save(yr,model,X,y,cols=0):

    yr=str(yr-1)

    # predictions=model.predict_proba(X.drop('PATIENT_ID',axis=1))
    # df=pd.DataFrame()
    # df['PATIENT_ID']=X['PATIENT_ID']
    # df['PRED']=[p[0] for p in predictions]
    # df['OBS']=y[config.COLUMNS[0]]
    from more_itertools import sliced
    CHUNK_SIZE = 50000
    
    index_slices = sliced(range(len(X)), CHUNK_SIZE)
    i=0
    n=len(X)/CHUNK_SIZE
    with open(config.PREDFILES[yr],'w') as predFile:
        csv_out=csv.writer(predFile)
        csv_out.writerow(['PATIENT_ID','PRED','OBS'])
        for index_slice in index_slices:
            i+=1
            util.vprint(i,'/',n)
            chunk = X.iloc[index_slice] # your dataframe chunk ready for use
            ychunk=y.iloc[index_slice]
            # print(all(chunk.PATIENT_ID==ychunk.PATIENT_ID))
            predictions=model.predict_proba(chunk.drop('PATIENT_ID',axis=1))[:,1] #Probab of being top1
            for element in zip(chunk['PATIENT_ID'],ychunk['PATIENT_ID'],predictions,ychunk[config.COLUMNS[0]]):
                csv_out.writerow(element)
                del element
    print('saved') 
def save(savefile,predfile,logistic,predictors,key='PROB_INGPROG'):
        import csv
        import pandas as pd
        with open(savefile,'w') as predFile:
            csv_out=csv.writer(predFile)
            csv_out.writerow(['PATIENT_ID',key])
            for chunk in pd.read_csv(predfile, 
                                      chunksize=10000):
                na_indices=chunk[chunk.isna().any(axis=1)].index
                chunk.drop(na_indices,axis=0,inplace=True)
                # chunk=chunk[['PATIENT_ID']+predictors]
                chunk.reset_index(inplace=True)
                # chunk['algunIngresoProg']=False
                # chunk.iloc[chunk.loc[chunk['ingresoProg']>0].index, chunk.columns.get_loc('algunIngresoProg')]=True
                predictions=logistic.predict_proba(chunk[predictors].drop('PATIENT_ID',axis=1))[:,1] #Probab of being top1
                for element in zip(chunk['PATIENT_ID'],predictions):
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

#%%
if __name__=='__main__':
        
    # FIXME STRUCT KEYS TO INT, FIX GENERARTABLASVARIABLES.RETRIEVE INDIICE
    configname='configurations/local/used/logistic20211203_135307.json'
    configuration=configure(configname)
    modelfilename='models/OLDBASE/logistic20211203_135307.joblib'
    model=job.load(modelfilename)
    
    
    X,y=getData(2017,prediction=True,oldbase=False)
    Xx,Yy=getData(2017,prediction=True,oldbase=True)
    cols=list(X.columns)
    if 'ingresoUrg' in cols: cols.remove('ingresoUrg')
    
    save('test2018.csv',
          config.DATAPATH+'test{0}.csv'.format(2017),
          model,cols,key='PROB')   
    # #%%
    probs=pd.read_csv('test2018.csv')
    # #%%
    # dat['binurg']=np.where(dat.urg>=1,1,0)
    auc=roc_auc_score(np.where(y.urg>=1,1,0), probs.PROB)
    oldprobs=pd.read_csv('untracked/predecirIngresos/logistica/urgSinPrevio18.csv')
    oldauc=roc_auc_score(np.where(Yy>=1,1,0), oldprobs.PROB_ING)

