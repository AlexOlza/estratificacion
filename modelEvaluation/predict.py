#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:59:25 2021

@author: aolza
"""
#%%
#EXTERNAL LIBRARIES
from pathlib import Path
import joblib as job
import sys
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
import csv
import pandas as pd
from tensorflow import keras
import zipfile

#%%
msg='Full path to configuration json file'
import argparse
parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--year', type=int,default=argparse.SUPPRESS,
                    help='The year for which you want to compare the predictions.')
parser.add_argument('--config_used', type=str, default=argparse.SUPPRESS,
                    help=msg)


args, unknown_args = parser.parse_known_args()

# year=args.year


sys.path.append('/home/aolza/Desktop/estratificacion/')
from python_settings import settings as config
from configurations.utility import configure
if not config.configured: 
    config_used=Path(args.config_used) if hasattr(args, 'config_used') else Path(input(msg))
    configuration=configure(config_used,TRACEBACK=False, VERBOSE=True)
    model_name=config_used.stem
try:
    experiment_name=config.EXPERIMENT
except:
    experiment_name=input('EXPERIMENT NAME (example: urgcms_excl_nbinj): ')

from dataManipulation.dataPreparation import getData
import os
#%%     FUNCTIONS
def generate_filename(model_name,yr, calibrated=False):
    if calibrated:
        fname=os.path.join(config.PREDPATH,'{0}_calibrated_{1}.csv'.format(model_name,yr))
    else: 
        fname=os.path.join(config.PREDPATH,'{0}__{1}.csv'.format(model_name,yr))
    return fname
def predict_save(yr,model,model_name,X,y,**kwargs):
    columns=kwargs.get('columns',config.COLUMNS[0])
    verbose=kwargs.get('verbose',config.VERBOSE)
    predictors=kwargs.get('predictors',config.PREDICTORREGEX)
    filename=kwargs.get('filename',None)
    # X=X.filter(regex=predictors)
    # print(predictors, len(X.filter(regex=predictors).columns))
    from more_itertools import sliced
    CHUNK_SIZE = 50000 #TODO experiment with this to try to speed up prediction
    
    index_slices = sliced(range(len(X)), CHUNK_SIZE)
    i=0
    n=len(X)/CHUNK_SIZE
    filename=generate_filename(filename,yr,calibrated=False) if not filename else filename

    if 'neural' in model_name:
        pred=model.predict
    else:
        pred=model.predict_proba
    with open(filename,'w') as predFile:
        csv_out=csv.writer(predFile)
        csv_out.writerow(['PATIENT_ID','PRED','OBS'])
        for index_slice in index_slices:
            if verbose:
                i+=1
                print(i,'/',n)
            chunk = X.iloc[index_slice] # your dataframe chunk ready for use
            ychunk=y.iloc[index_slice]
            if 'COSTE_TOTAL_ANO2' in config.COLUMNS:
                predictions=model.predict(chunk.drop('PATIENT_ID',axis=1)) # predicted cost
            else:
                if 'neural' in model_name:
                    predictions=pred(chunk.drop('PATIENT_ID',axis=1))[:,0] #Probab of hospitalization (Keras)
                else:
                    predictions=pred(chunk.drop('PATIENT_ID',axis=1))[:,1] #Probab of hospitalization (sklearn)
            for element in zip(chunk['PATIENT_ID'],ychunk['PATIENT_ID'],predictions,ychunk[columns]):
                csv_out.writerow(element)
                del element

    print('saved',filename) 
def to_zip(filename):
    zipfilename = '/'.join(filename.split('/')[:-1])+'.zip'
    mode='a' if zipfile.is_zipfile(zipfilename) else 'w'
    zfile= zipfile.ZipFile(zipfilename, mode=mode)
    zfile.write(filename, arcname=filename.split('/')[-1])
    zfile.close()
    print('saved in zip ',zipfilename)
import re
def predict(model_name,experiment_name,year,**kwargs):
    predictors=kwargs.get('predictors',config.PREDICTORREGEX)
    filename=kwargs.get('filename',model_name)
    modelfilename=os.path.join(re.sub(config.EXPERIMENT,experiment_name,config.MODELPATH),model_name)
    if 'neural' in model_name:
        load=keras.models.load_model
        if model_name=='neural_AGESEX':
            predictors=r'PATIENT_ID|FEMALE|AGE'
    else:
        modelfilename+='.joblib'
        load=job.load
    if Path(modelfilename):
        print('loading model ',model_name)
        model=load(modelfilename)
    else:
        print('Model not found :(')
        print('missing ',modelfilename)
        return (None, None)
    Xx=kwargs.get('X',None)
    Yy=kwargs.get('y',None)
    if (not isinstance(Xx,pd.DataFrame)) or (not isinstance(Yy,pd.DataFrame)):
        Xx,Yy=getData(year-1,predictors=predictors)

    predFilename=generate_filename(filename,year)
    calibFilename=generate_filename(filename,year, calibrated=True)
    zipfilename = '/'.join(predFilename.split('/')[:-1])+'.zip'
    #Conditions
    calibrated_predictions_found= Path(calibFilename).is_file()
    uncalibrated_predictions_found= Path(predFilename).is_file()
    no_predictions_found=(not uncalibrated_predictions_found) and (not calibrated_predictions_found)
    zipfile_found=zipfile.is_zipfile(zipfilename)
    
    if zipfile_found:
        zfile=zipfile.ZipFile(zipfilename,'r')
        zipfile_contains_calibrated=os.path.basename(calibFilename) in zfile.namelist()
        zipfile_contains_uncalibrated=os.path.basename(predFilename) in zfile.namelist()
        if zipfile_contains_calibrated:
            print('Calibrated predictions found; loading from zip')           
            probs=pd.read_csv(zfile.open(os.path.basename(calibFilename))) 
            probs=probs[['PATIENT_ID', 'PRED', 'OBS']]
            if 'COSTE_TOTAL_ANO2' in config.COLUMNS:
                score=r2_score(probs.OBS, probs.PRED)
            else:
                score=roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)
            print('score ',score) 
            return (probs,score)
        elif zipfile_contains_uncalibrated:
            print('Uncalibrated predictions found; loading from zip')   
            probs=pd.read_csv(zfile.open(os.path.basename(predFilename))) 
            if 'COSTE_TOTAL_ANO2' in config.COLUMNS:
                score=r2_score(probs.OBS, probs.PRED)
            else:
                score=roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)
            print('score ',score) 
            return (probs,score)
        else:
            print('Zip file does not contain the wanted predictions')

    
    if  no_predictions_found:
        print('No predictions found')
        predict_save(year, model,model_name, Xx, Yy, 
                     filename=predFilename,
                     predictors=predictors, verbose=False)
        uncalibrated_predictions_found=True
    if calibrated_predictions_found:
        print('Calibrated predictions found; loading')
        probs=pd.read_csv(calibFilename) 
        # probs=probs[['PATIENT_ID', 'PRED', 'OBS']]
        to_zip(calibFilename)
    elif uncalibrated_predictions_found: 
        print('Uncalibrated predictions found; loading')
        probs=pd.read_csv(predFilename) 
    if 'COSTE_TOTAL_ANO2' in config.COLUMNS:
        score=r2_score(probs.OBS, probs.PRED)
    else:
        score=roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PRED)
    print('score ',score) 
    return (probs,score)
#%%
if __name__=='__main__':
        
    year=int(input('YEAR YOU WANT TO PREDICT:'))
    probs,score=predict(model_name,experiment_name,year)       
