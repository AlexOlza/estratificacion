#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
import numpy as np
# np.random.seed(42)
# rng = np.random.RandomState(42)
# rng.seed(42)
# print(rng.random_sample(5))
# print(np.random.randint(0,100,size=5))
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.utility import configure
import joblib as job
configname='/home/aolza/Desktop/estratificacion/configurations/used/randomForest20211231_114533.json'
configuration=configure(configname,TRACEBACK=False, VERBOSE=True)

    #%%
import csv
import pandas as pd
import configurations.utility as util
from dataManipulation.dataPreparation import getData
from sklearn.metrics import roc_auc_score
# rng = np.random.RandomState(42)

# np.random.seed(config.SEED) #This NOT is enough for intra-machine reproducibility
# seed= np.random.default_rng(42)
# print('the seed has ben set rng')
#%%
if __name__=='__main__':
        
    # FIXME STRUCT KEYS TO INT, FIX GENERARTABLASVARIABLES.RETRIEVE INDICE
    
    modelfilename='/home/aolza/Desktop/estratificacion/models/urgcms_excl_hdia_nbinj/randomForest20211231_114533.joblib'

    model=job.load(modelfilename)
    # model=model.set_params(n_jobs=1, random_state=42)#NOT enough for intra-machine reproducibility
    print('njobs ',model.n_jobs)
    print('rand state',model.random_state)

    Xx,Yy=getData(2017)
    print('min {0}='.format(config.COLUMNS[0]),min(Yy[config.COLUMNS[0]]))#FIXME remove this
    # x,y=getData(2017,oldbase=True)
    cols=list(Xx.columns)
    if 'ingresoUrg' in cols: cols.remove('ingresoUrg')
    d={}
    x=Xx.head()
    y=Yy.head()
    # x=x.reindex(sorted(x.columns), axis=1)
    for i in range(5):
        d[i]=[]
        print('patient id','pred','obs')
        preds=model.predict_proba(x.drop('PATIENT_ID',axis=1))[:,1] 
        for element in zip(x['PATIENT_ID'],y['PATIENT_ID'],preds,y[config.COLUMNS[0]]):
                print(element)
                d[i].append(element)
                

    print('any different predictions? ',np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).any())
    print('max diff, min diff: ', np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).max(),np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).min())
    # print('auc ',roc_auc_score(np.where(Yy.head()[config.COLUMNS[0]]>=1,1,0), preds)) 
    # oldprobs=pd.read_csv('untracked/predecirIngresos/logistica/urgSinPrevio18.csv')
    # oldauc=roc_auc_score(np.where(y.ingresoUrg>=1,1,0), oldprobs.PROB_ING)

    # print(model.score(Xx.drop('PATIENT_ID',axis=1),Yy.drop('PATIENT_ID',axis=1))) 
"""
    cluster=[(1000002979, 0.6592525474680007, 0.0)
(100000795, 0.2935369763176843, 0.0)
(1000012982, 0.6574054066504159, 0.0)
(100004640, 0.2967072866031834, 0.0)
(1000055337, 0.29046298165051154, 0.0)]
    
    clusternoseed=[(1000002979, 0.474291785655645, 0.0)
(100000795, 0.37100028446290256, 0.0)
(1000012982, 0.43264545858871967, 0.0)
(100004640, 0.2906317606087574, 0.0)
(1000055337, 0.29046298165051154, 0.0)
]
    clusternoseed=(1000002979, 0.5284741349226243, 0.0)
(100000795, 0.3413538406822807, 0.0)
(1000012982, 0.3244141079827993, 0.0)
(100004640, 0.2909499769754741, 0.0)
(1000055337, 0.29046298165051154, 0.0)

LOCAL: 
    WITH the seed has ben set, njobs  -1: max diff, min diff:  5.551115123125783e-17 -5.551115123125783e-17
    diff between executions
    
    WITH njobs  1, no seed: no diff same exec, diff between exec
    
    WITH njobs 1, seed set: no diff same exec, diff between exec
    
    WITH njobs 1, seed rng set: no diff same exec, diff between exec
        
        seems like never changes (1000055337, 0.29046298165051154, 0.0)
"""