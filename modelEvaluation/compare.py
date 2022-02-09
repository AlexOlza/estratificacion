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


def update_all_preds(all_predictions):
    #Save if necessary
    all_preds='{0}/all_preds.csv'.format(config.PREDPATH) #Filename
    if Path('{0}/all_preds.csv'.format(config.PREDPATH)).is_file():
        print('all_preds.csv located')
        saved=pd.read_csv(all_preds,nrows=3)
        if not set(saved.columns)==set(all_predictions.columns):
            print('Adding new columns')
            saved=saved=pd.read_csv(all_preds)
            all_predictions=pd.merge(all_predictions,saved,on=['PATIENT_ID','OBS'],how='inner')
            all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
    else:
        all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
        print('Saved ' '{0}/all_preds.csv'.format(config.PREDPATH))

year=int(input('YEAR YOU WANT TO PREDICT: '))
assert year in [2017,2018,2019], 'No data available!'

available_models=[x.stem for x in Path(config.MODELPATH).glob('**/*') if x.is_file()]
experiment_name=Path(config.MODELPATH).parts[-1]
print('Available models are:')
print(available_models)

print('Loading latest models per algorithm:')
ids = [int(''.join(re.findall('\d+',model))) for model in available_models]
algorithms=['_'.join(re.findall('[^\d+_\d+]+',model)) for model in available_models]
df=pd.DataFrame(list(zip(algorithms,ids,[i for i in range(len(ids))])),columns=['algorithm','id','i'])
selected=df.loc[df.algorithm!='nested_log'].groupby(['algorithm']).apply(lambda x: x.loc[x.id == x.id.max()].i).to_numpy()
selected=[available_models[i] for i in selected]
print(selected)
X,y=getData(year-1)
aucs=[]
all_predictions=pd.DataFrame()
for m in selected:
    probs,auc=predict(m,experiment_name,year,X=X,y=y)
    aucs.append(auc)
    try:
        all_predictions=pd.merge(all_predictions,probs,on=['PATIENT_ID','OBS'],how='inner')
    except:
        all_predictions=probs
    all_predictions.rename(columns={'PRED': 'PRED_{0}'.format(m)}, inplace=True)

pd.set_option('display.max_columns',len(selected)+2) #show all columns
print('AUCS ',aucs)
print('Predictions')
print(all_predictions.head())
