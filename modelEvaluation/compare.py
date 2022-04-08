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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score, brier_score_loss,RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import os
import pandas as pd
from pathlib import Path
import re
import joblib as job
import json
import argparse

parser = argparse.ArgumentParser(description='Compare models')
parser.add_argument('--year', '-y', type=int,default=argparse.SUPPRESS,
                    help='The year for which you want to compare the predictions.')
parser.add_argument('--nested','-n', dest='nested', action='store_true', default=False,
                    help='Are you comparing nested models with the same algorithm?')
parser.add_argument('--all','-a', dest='all', action='store_true', default=True,
                    help='Compare all models with the same algorithm?')
parser.add_argument('--config_used', type=str, default=argparse.SUPPRESS,
                help='Full path to configuration json file: ')

args = parser.parse_args()

from python_settings import settings as config

if not config.configured: 
    import configurations.utility as util
    config_used=args.config_used if hasattr(args, 'config_used') else os.path.join(os.environ['USEDCONFIG_PATH'],input('Experiment...'),input('Model...')+'.json')
    configuration=util.configure(config_used,TRACEBACK=True, VERBOSE=True)
import configurations.utility as util
from modelEvaluation.predict import predict, generate_filename
from dataManipulation.dataPreparation import getData
from modelEvaluation.detect import detect_models, detect_latest
from modelEvaluation.calibrate import calibrate
#%%
def compare_nested(available_models,X,y,year):
    available_models=[m for m in available_models if ('nested' in m)]
    available_models.sort()
    variable_groups=[r'PATIENT_ID|FEMALE|AGE_[0-9]+$','EDC_','RXMG_','ACG']
    predictors={}
    for i in range(1,len(variable_groups)+1):
        predictors[available_models[i-1]]=r'|'.join(variable_groups[:i])
    metrics=compare(available_models,X,y,year,predictors=predictors)
    return(metrics)

def compare(selected,X,y,year,experiment_name=Path(config.MODELPATH).parts[-1],**kwargs):
    import traceback
    K=kwargs.get('K',20000)
    predictors=kwargs.get('predictors',{m:config.PREDICTORREGEX for m in selected})
    metrics={'Score':{},f'Recall_{K}':{},f'PPV_{K}':{}, 'Brier':{}}
    for m in selected:
        try:
            probs=calibrate(m,year,experiment_name=experiment_name,presentX=X,presentY=y,predictors=predictors[m])
            if (probs is None):#If model not found
                continue
            metrics['Score'][m]=roc_auc_score(np.where(probs.OBS>=1,1,0), probs.PREDCAL)
            metrics[f'Recall_{K}'][m],metrics[f'PPV_{K}'][m], _, _=performance(np.where(probs.OBS>=1,1,0), probs.PREDCAL,K)
            metrics['Brier'][m]=brier_score_loss(np.where(probs.OBS>=1,1,0), probs.PREDCAL)
        except Exception as exc:
            print('Something went wrong for model ', m)
            print(traceback.format_exc())
            print(exc)

    return(metrics)

import numpy as np
from sklearn.metrics import confusion_matrix
def performance(obs,pred,K): 
    orderedPred=sorted(pred,reverse=True)
    orderedObs=sorted(obs,reverse=True)
    cutoff=orderedPred[K-1]
    print(f'Cutoff value ({K} values): {cutoff}')
    print(f'Observed cutoff value ({K} values): {orderedObs[K-1]}')
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
    specificity = tn / (tn+fp)
    print('Recall, PPV, Spec = ',recall,ppv, specificity)
    return(recall,ppv, specificity, newpred)

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

def boxplots(df, year, K, parent_metrics=None, parentNeural=False, **kwargs):
    path=kwargs.get('path',config.FIGUREPATH)
    name=kwargs.get('name','')
    X=kwargs.get('X',None)
    y=kwargs.get('y',None)
    if not parent_metrics: #we have to compute them
        parent_models=[i for i in detect_models(re.sub('hyperparameter_variability_|fixsample_','',config.MODELPATH))  if 'logistic' in i]
        logistic_model=[i for i in detect_latest(parent_models) if 'logistic2022' in i][0]
        # logistic_predfile=re.sub('hyperparameter_variability_|fixsample_','',generate_filename(logistic_model,year, calibrated=True))
        # logistic_predictions=pd.read_csv(logistic_predfile)
        logistic_predictions=calibrate(logistic_model, year,experiment_name=re.sub('hyperparameter_variability_|fixsample_','',config.EXPERIMENT), presentX=X, presentY=y)
        auc=roc_auc_score(np.where(logistic_predictions.OBS>=1,1,0), logistic_predictions.PRED)
        brier=brier_score_loss(np.where(logistic_predictions.OBS>=1,1,0), logistic_predictions.PREDCAL)
        recall, ppv, _, _= performance(logistic_predictions.OBS,logistic_predictions.PRED, K)
        parent_metrics={'Model':[logistic_model],
                        'Score':[auc],
                        f'Recall_{K}': [recall],
                        f'PPV_{K}':[ppv],
                        'Brier':[brier]}
    parent_df=pd.DataFrame.from_dict(parent_metrics)   
    parent_metrics[f'F1_{K}'] = 2*parent_df[f'Recall_{K}']*parent_df[f'PPV_{K}']/(parent_df[f'Recall_{K}']+parent_df[f'PPV_{K}'])
    df['Algorithm']=[re.sub('_|[0-9]', '', model) for model in df['Model'].values]
    df[f'F1_{K}']=2*df[f'Recall_{K}']*df[f'PPV_{K}']/(df[f'Recall_{K}']+df[f'PPV_{K}'])
   
    for column in ['Score', f'Recall_{K}', f'PPV_{K}','Brier',f'F1_{K}']:
        print(column)
        fig, ax = plt.subplots(figsize=(8,12))
        plt.suptitle('')

        df.boxplot(column=column, by='Algorithm', ax=ax)
        for model, value in zip(parent_metrics['Model'], parent_metrics[column]):
            if parentNeural:
                if any(['logistic' in model,'neural' in model]): #exclude other algorithms
                    plt.axhline(y = value, linestyle = '-', label=model, color=next(ax._get_lines.prop_cycler)['color'])
            else:
                if any(['logistic' in model]): #exclude other algorithms
                    plt.axhline(y = value, linestyle = '-', label=model, color='r')
        plt.legend()
        plt.savefig(os.path.join(path,f'hyperparameter_variability_{column}.png'))
        plt.show()

def roc_pr_curves(modelDict, yr, parent_model, **kwargs):
    # load logistic predictions
    parent=calibrate(parent_model, yr)
    models, roc, pr={},{},{}
    # load models predictions
    for label,  model in modelDict.items(): 
        print( label,  model)
        models[label]=calibrate(model, yr)
        obs=np.where(models[label].OBS>=1,1,0)
        fpr, tpr, _ = roc_curve(obs, models[label].PREDCAL)
        prec, rec, _ = precision_recall_curve(obs, models[label].PREDCAL)
        roc_auc = auc(fpr, tpr)
        avg_prec = average_precision_score(obs, models[label].PREDCAL)
        pr[label]=PrecisionRecallDisplay(prec, rec, 
                                     estimator_name=label,
                                     average_precision=avg_prec)
        roc[label]= RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                      estimator_name=label)
    return(roc, pr)


def model_percentiles(df,metric,plot,*args, **kwargs):
    def q1(x):
        return x.quantile(0.05,interpolation='nearest')

    def q2(x):
        return x.quantile(0.5,interpolation='nearest')
    
    def q3(x):
        return x.quantile(0.95,interpolation='nearest')

    brierdist=df.groupby(['Algorithm'])[metric].agg([ q1, q2, q3]).stack(level=0)
    print(f'{metric} distribution per algorithm: ')
    print(brierdist)
    roc, pr={},{}
    low, median, high = {},{},{}
    models_to_plot={}
    selected_models=[]
    for alg in df.Algorithm.unique():
        df_alg=df.loc[df.Algorithm==alg].to_dict(orient='list')
        perc25=brierdist.loc[alg]['q1']
        perc50=brierdist.loc[alg]['q2']
        perc75=brierdist.loc[alg]['q3']
        low[alg]=list(df_alg['Model'])[list(df_alg[metric]).index(perc25)]
        median[alg]=list(df_alg['Model'])[list(df_alg[metric]).index(perc50)]
        high[alg]=list(df_alg['Model'])[list(df_alg[metric]).index(perc75)]
        # selected_models=[low[alg],median[alg],high[alg]]
        models_to_plot[alg]={'Perc. 05':low[alg],'Perc. 50':median[alg],'Perc. 95':high[alg]}
        roc[alg], pr[alg] = plot(models_to_plot[alg],*args, **kwargs)
    return(roc, pr)
#%%
if __name__=='__main__':
    year=int(input('YEAR TO PREDICT: ')) if not hasattr(args, 'year') else args.year
    nested=eval(input('NESTED MODEL COMPARISON? (True/False) ')) if not hasattr(args, 'nested') else args.nested
    all_models=eval(input('COMPARE ALL MODELS? (True/False) ')) if not hasattr(args, 'all') else args.all
    
    available_models=detect_models()
    
    if nested:   
        selected=sorted([m for m in available_models if ('nested' in m)])
    elif all_models:
        selected=[m for m in available_models if not ('nested' in m)]
    else:
        selected=detect_latest(available_models)
    
    if Path(config.PREDPATH+'/metrics.csv').is_file():
        available_metrics=pd.read_csv(config.PREDPATH+'/metrics.csv')
    else:
        available_metrics=pd.DataFrame.from_dict({'Model':[]})
    if all([s in available_metrics.Model.values for s in selected]):
        print('All metrics are available')
        print(available_metrics)
        available_metrics['Algorithm']=[re.sub('_|[0-9]', '', model) for model in available_metrics['Model'].values]
        print(available_metrics.groupby('Algorithm').describe().transpose())
        # parent_metrics=pd.read_csv(re.sub('hyperparameter_variability_|fixsample_','',config.PREDPATH+'/metrics.csv')).to_dict('list')
        boxplots(available_metrics, year, K=20000)
    else:
        selected=[s for s in selected if not (s in available_metrics.Model.values)]
        
        X,y=getData(year-1)

        if not nested:
            metrics=compare(selected,X,y,year)

        if nested:
            metrics=compare_nested(available_models,X,y,year)
            variable_groups=[r'SEX+ AGE','+ EDC_','+ RXMG_','+ ACG']
            score,recall,ppv, brier=[list(array.values()) for array in list(metrics.values())]
            print(pd.DataFrame(list(zip(selected,variable_groups,score,recall,ppv)),columns=['Model','Predictors']+list(metrics.keys())).to_markdown(index=False))
        else:
            score,recall,ppv, brier=[list(array.values()) for array in list(metrics.values())]
            df=pd.DataFrame(list(zip(selected,score,recall,ppv, brier)),columns=['Model']+list(metrics.keys()))
            print(df.to_markdown(index=False,))
        
        df=pd.concat([df, available_metrics], ignore_index=True, axis=0)
        df.to_csv(config.PREDPATH+'/metrics.csv', index=False)
    
        # parent_metrics=pd.read_csv(re.sub('hyperparameter_variability_|fixsample_','',config.PREDPATH+'/metrics.csv')).to_dict('list')
        boxplots(df, year, K=20000, X=X, y=y)
        print(df.groupby('Algorithm').describe().transpose())
        
    algorithms=['randomForest', 'hgb', 'neuralNetworkRandomCLR']
    parent_models=[i for i in detect_models(re.sub('hyperparameter_variability_|fixsample_','',config.MODELPATH))  if 'logistic' in i]
    logistic_model=[i for i in detect_latest(parent_models) if 'logistic2022' in i][0]
        
    roc, pr= model_percentiles(df, 'Score', roc_pr_curves, 2018, logistic_model)
    
    fig1, (ax11,ax12, ax13) = plt.subplots(1,3,figsize=(16,8))
    fig2, (ax21,ax22, ax23) = plt.subplots(1,3,figsize=(16,8))
    rocaxes=(ax11,ax12, ax13)
    praxes=(ax21,ax22, ax23)
    for alg, axroc, axpr in zip(algorithms, rocaxes, praxes):
        for perc in ['Perc. 05', 'Perc. 50', 'Perc. 95']:
            roc[alg][perc].plot(axroc)
            pr[alg][perc].plot(axpr)
        axroc.set_title(alg)
        axpr.set_title(alg)
