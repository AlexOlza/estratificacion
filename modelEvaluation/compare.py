#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:13:19 2021

@author: aolza


    Input: Experiment name, prediction year
    In the models/experiment directory, detect...
        All algorithms present
        The latest model for each algorithm
    Prompt user for consent about the selected models
    Load models
    Predict (if necessary, i.e. if config.PREDPATH+'/{0}__{1}.csv'.format(model_name,yr) is not found )
    Save into big dataframe (not sure this is clever)
    Calibrate (coming soon)
    Compare

"""

import pandas as pd
from pathlib import Path
import re

from python_settings import settings as config
import configurations.utility as util
from modelEvaluation.predict import predict, generate_filename
util.configure('configurations.local.logistic')

from dataManipulation.dataPreparation import getData

def detect_models():
    available_models=[x.stem for x in Path(config.MODELPATH).glob('**/*') if x.is_file()]
    print('Available models are:')
    print(available_models)
    return(available_models)

def load_latest(available_models):      
    print('Loading latest models per algorithm:')
    ids = [int(''.join(re.findall('\d+',model))) for model in available_models]
    algorithms=['_'.join(re.findall('[^\d+_\d+]+',model)) for model in available_models]
    df=pd.DataFrame(list(zip(algorithms,ids,[i for i in range(len(ids))])),columns=['algorithm','id','i'])
    selected=df.loc[df.algorithm!='nested_log'].groupby(['algorithm']).apply(lambda x: x.loc[x.id == x.id.max()].i).to_numpy()
    selected=[available_models[i] for i in selected]
    print(selected)
    return(selected)

def compare_nested(available_models,X,y,year):
    available_models=[m for m in available_models if ('nested' in m)]
    available_models.sort()
    variable_groups=[r'PATIENT_ID|FEMALE|AGE_[0-9]+$','EDC_','RXMG_','ACG']
    predictors={}
    for i in range(1,len(variable_groups)+1):
        predictors[available_models[i-1]]=r'|'.join(variable_groups[:i])
    return(compare(available_models,X,y,year,predictors=predictors))

def compare(selected,X,y,year,experiment_name=Path(config.MODELPATH).parts[-1],**kwargs):
    predictors=kwargs.get('predictors',{m:config.PREDICTORREGEX for m in selected})
    aucs=[]
    all_predictions=pd.DataFrame()
    for m in selected:
        probs,auc=predict(m,experiment_name,year,X=X,y=y,predictors=predictors[m])
        aucs.append(auc)
        try:
            all_predictions=pd.merge(all_predictions,probs,on=['PATIENT_ID','OBS'],how='inner')
        except:
            all_predictions=probs
        all_predictions.rename(columns={'PRED': 'PRED_{0}'.format(m)}, inplace=True)
    return(all_predictions)
def update_all_preds(all_predictions,selected):
    #Save if necessary
    all_preds='{0}/all_preds.csv'.format(config.PREDPATH) #Filename
    if Path('{0}/all_preds.csv'.format(config.PREDPATH)).is_file():
        print('all_preds.csv located')
        saved=pd.read_csv(all_preds,nrows=3)
        if not set(saved.columns)==set(all_predictions.columns):
            print('Adding new columns')
            saved=pd.read_csv(all_preds)
            all_predictions=pd.merge(all_predictions,saved,on=['PATIENT_ID','OBS'],how='inner')
            all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
    else:
        all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
        print('Saved ' '{0}/all_preds.csv'.format(config.PREDPATH))


    pd.set_option('display.max_columns',len(selected)+2) #show all columns
    print('AUCS ',aucs)
    print('Predictions')
    print(all_predictions.head())

def main(year=2018,nested=False):
    X,y=getData(year-1)
    available_models=detect_models()
    if nested:
        all_predictions=compare_nested(available_models,X,y,year)
        selected=available_models
    else:
        selected=load_latest(available_models)
        all_predictions=compare(selected,X,y,year)
    update_all_preds(all_predictions,selected)
        
if __name__=='__main__':
    main(nested=True)


