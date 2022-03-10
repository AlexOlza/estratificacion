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
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd
from pathlib import Path
import re
import joblib as job
import json
import argparse

parser = argparse.ArgumentParser(description='Compare models')
parser.add_argument('--year', '-y', type=int,default=argparse.SUPPRESS,
                    help='The year for which you want to compare the predictions.')
parser.add_argument('--nested','-n', type=bool, default=argparse.SUPPRESS,
                    help='Are you comparing nested models with the same algorithm?')
parser.add_argument('--all','-a',type=bool, default=argparse.SUPPRESS,
                    help='Compare all models with the same algorithm?')
parser.add_argument('--config_used', type=str, default=argparse.SUPPRESS,
                help='Full path to configuration json file')

args = parser.parse_args()
print(vars(args))
config_used=args.config_used if hasattr(args, 'config_used') else input('Full path to configuration json file')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used,TRACEBACK=True, VERBOSE=True)

from modelEvaluation.predict import predict, generate_filename
from dataManipulation.dataPreparation import getData
#%%
def detect_models():
    available_models=[x.stem for x in Path(config.MODELPATH).glob('**/*') if x.is_file()]
    print('Available models are:')
    print(available_models)
    return(available_models)

def detect_latest(available_models): 
    if len(available_models)==1:
        print('There is only one model')
        return(available_models)
    print('Selecting latest models per algorithm:')
    ids = [int(''.join(re.findall('\d+',model))) for model in available_models]
    algorithms=['_'.join(re.findall('[^\d+_\d+]+',model)) for model in available_models]
    df=pd.DataFrame(list(zip(algorithms,ids,[i for i in range(len(ids))])),columns=['algorithm','id','i'])
    selected=df.loc[~(df.algorithm.str.startswith('nested'))].groupby(['algorithm']).apply(lambda x: x.loc[x.id == x.id.max()].i).to_numpy()
    selected=[available_models[i] for i in selected.ravel()]
    print(selected)
    return(selected)

def compare_nested(available_models,X,y,year):
    available_models=[m for m in available_models if ('nested' in m)]
    available_models.sort()
    variable_groups=[r'PATIENT_ID|FEMALE|AGE_[0-9]+$','EDC_','RXMG_','ACG']
    predictors={}
    for i in range(1,len(variable_groups)+1):
        predictors[available_models[i-1]]=r'|'.join(variable_groups[:i])
    all_predictions,metrics=compare(available_models,X,y,year,predictors=predictors)
    return(all_predictions,metrics)

def compare(selected,X,y,year,experiment_name=Path(config.MODELPATH).parts[-1],**kwargs):
    K=kwargs.get('K',20000)
    predictors=kwargs.get('predictors',{m:config.PREDICTORREGEX for m in selected})
    metrics={'Score':{},'Recall_{0}'.format(K):{},'PPV_{0}'.format(K):{}}
    all_predictions=pd.DataFrame()
    for m in selected:
        probs,auc=predict(m,experiment_name,year,X=X,y=y,predictors=predictors[m])
        if (probs is None) and (auc is None):#If model not found
            continue
        metrics['Score'][m]=auc
        try:  
            all_predictions=pd.merge(all_predictions,probs,on=['PATIENT_ID','OBS'],how='inner')
        except:
            all_predictions=probs
        all_predictions.rename(columns={'PRED': 'PRED_{0}'.format(m)}, inplace=True)
        metrics['Recall_{0}'.format(K)][m],metrics['PPV_{0}'.format(K)][m]=performance(all_predictions['PRED_{0}'.format(m)],all_predictions.OBS,K)
        
        # selected=[m for m in available_models if ('nested' in m)]
        # selected.sort()
    return(all_predictions,metrics)
def update_all_preds(all_predictions,selected):
    #Save if necessary
    all_preds='{0}/all_preds.csv'.format(config.PREDPATH) #Filename
    if Path('{0}/all_preds.csv'.format(config.PREDPATH)).is_file():
        print('all_preds.csv located')
        saved=pd.read_csv(all_preds,nrows=3)
        common_cols=list(set(saved.columns).intersection(set(all_predictions.columns)))
        if (set(all_predictions.columns)-set(saved.columns)):#if any of our columns is not already saved
            print('Adding new columns')
            saved=pd.read_csv(all_preds)
            all_predictions=pd.merge(all_predictions,
                                     saved,
                                     on=common_cols,
                                     how='inner')
            all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
        else:
            print('No new columns to add.')
    else:
        all_predictions.to_csv('{0}/all_preds.csv'.format(config.PREDPATH),index=False)
        print('Saved ' '{0}/all_preds.csv'.format(config.PREDPATH))
    return(all_predictions)
import numpy as np
from sklearn.metrics import confusion_matrix
def performance(pred,obs,K): 
    orderedPred=sorted(pred,reverse=True)
    orderedObs=sorted(obs,reverse=True)
    cutoff=orderedPred[K-1]
    print('Cutoff value ({0} values): {1}'.format(K, cutoff))
    print('Observed cutoff value ({0} values): {1}'.format(K, orderedObs[K-1]))
    newpred=pred>=cutoff
    print('Length of selected list ',sum(newpred))
    if 'COSTE_TOTAL_ANO2' in config.COLUMNS: #maybe better: not all([int(i)==i for i in obs])
        newobs=obs>=orderedObs[K-1]
    else:
        newobs=np.where(obs>=1,1,0) #Whether the patient had ANY admission 
    c=confusion_matrix(y_true=newobs, y_pred=newpred)
    print(c)
    tn, fp, fn, tp =c.ravel()
    print(' tn, fp, fn, tp =',tn, fp, fn, tp)
    recall=c[1][1]/(c[1][0]+c[1][1])
    ppv=c[1][1]/(c[0][1]+c[1][1])
    print('Recall, Positive Predictive Value = ',recall,ppv)
    return(recall,ppv)

def parameter_distribution(models,**args):
    model_dict,grid,params,times_selected={},{},{},{}
    for m in models:
        print(m)
        try:
            grid[m]=json.load(open(config.USEDCONFIGPATH+config.EXPERIMENT+'/hgb_19.json'))["RANDOM_GRID"]
        except KeyError:
            print('No RANDOM_GRID found in config :(')
        try:
            model_dict[m]=job.load(config.MODELPATH+m+'.joblib')
            params[m]=model_dict[m].get_params()
        except FileNotFoundError:
            print('Model not found :(')
    print(params)
    print(model_dict)
    for parameter,options in grid[m].items():
        times_selected[parameter]={}
        for m in models:
            opt=params[m][parameter]
            try:
                times_selected[parameter][opt]+=1
            except KeyError:
                times_selected[parameter][opt]=1
    for parameter in times_selected.keys():
        times_selected[parameter]=pd.DataFrame(times_selected[parameter],
                                                         index=[0]) #fixes ValueError: If using all scalar values, you must pass an index
        plt.figure()
        times_selected[parameter].plot(kind='bar',title=parameter, rot=0)
def performance_distribution(models):
    pass #TODO  write
#%%
if __name__=='__main__':
    year=int(input('YEAR TO PREDICT: ')) if not hasattr(args, 'year') else args.year
    nested=eval(input('NESTED MODEL COMPARISON? (True/False) ')) if not hasattr(args, 'nested') else args.nested
    all_models=eval(input('COMPARE ALL MODELS? (True/False) ')) if not hasattr(args, 'all') else args.all
    X,y=getData(year-1)
    available_models=detect_models()
    if nested:
        all_predictions,metrics=compare_nested(available_models,X,y,year)
        selected=sorted([m for m in available_models if ('nested' in m)])
    elif all_models:
        all_predictions,metrics=compare(available_models,X,y,year)
        selected=available_models
    else:
        selected=detect_latest(available_models)
        all_predictions,metrics=compare(selected,X,y,year)
    all_predictions=update_all_preds(all_predictions,selected)

    if nested:
        variable_groups=[r'SEX+ AGE','+ EDC_','+ RXMG_','+ ACG']
        score,recall,ppv=[list(array.values()) for array in list(metrics.values())]
        print(pd.DataFrame(list(zip(selected,variable_groups,score,recall,ppv)),columns=['Model','Predictors']+list(metrics.keys())).to_markdown(index=False))
    else:
        variable_groups=['']*len(selected)
        score,recall,ppv=[list(array.values()) for array in list(metrics.values())]
        df=pd.DataFrame(list(zip(selected,variable_groups,score,recall,ppv)),columns=['Model','Predictors']+list(metrics.keys()))
        print(df.to_markdown(index=False,))
    ax=[0]*len(metrics)
    for i,m in enumerate(list(metrics.keys())):
        ax[i]=df[m].plot.box()
        # print(i,m)
    import matplotlib.pyplot as plt
    
    for column in df.select_dtypes(exclude=['object']):
        plt.figure()
        # df.boxplot([column])
        df[column].plot(kind='box',title=' - '.join([config.ALGORITHM,config.EXPERIMENT,column]))
    parameter_distribution(selected)
