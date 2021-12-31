#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
import numpy as np
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.utility import configure
import joblib as job
configname='/home/aolza/Desktop/estratificacion/configurations/used/randomForest20211231_114533.json'
configuration=configure(configname,TRACEBACK=False, VERBOSE=True)
from dataManipulation.dataPreparation import getData


#%%
if __name__=='__main__':
      
    modelfilename='/home/aolza/Desktop/estratificacion/models/urgcms_excl_hdia_nbinj/randomForest20211231_114533.joblib'

    model=job.load(modelfilename)

    Xx,Yy=getData(2017)

    d={}
    x=Xx.head()
    y=Yy.head()

    for i in range(5):
        d[i]=[]
        print('patient id','pred','obs')
        preds=model.predict_proba(x.drop('PATIENT_ID',axis=1))[:,1] 
        for element in zip(x['PATIENT_ID'],y['PATIENT_ID'],preds,y[config.COLUMNS[0]]):
                print(element)
                d[i].append(element)
                

    print('any different predictions? ',np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).any())
    print('max diff, min diff: ', np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).max(),np.array([np.array(d[i])-np.array(list(d.values()))[0] for i in d]).min())
