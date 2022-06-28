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
    Calibrate
    Compare

"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, RocCurveDisplay, roc_curve, auc, \
    precision_recall_curve, PrecisionRecallDisplay
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
parser.add_argument('--year', '-y', type=int, default=argparse.SUPPRESS,
                    help='The year for which you want to compare the predictions.')
parser.add_argument('--nested', '-n', dest='nested', action='store_true', default=False,
                    help='Are you comparing nested models with the same algorithm?')
parser.add_argument('--all', '-a', dest='all', action='store_true', default=True,
                    help='Compare all models with the same algorithm?')
parser.add_argument('--config_used', type=str, default=argparse.SUPPRESS,
                    help='Full path to configuration json file: ')

args = parser.parse_args()

from python_settings import settings as config

if not config.configured:
    import configurations.utility as util

    if hasattr(args, 'config_used'):
        config_used = args.config_used
    else:
        experiment = input('Experiment...')
        print('Available models:')
        p = Path(os.path.join(os.environ['USEDCONFIG_PATH'], experiment)).glob('**/*.json')
        files = [x.stem for x in p if x.is_file()]
        print(files)
        model = input('Model...')
        config_used = os.path.join(os.environ['USEDCONFIG_PATH'], experiment, model + '.json')
    configuration = util.configure(config_used, TRACEBACK=True, VERBOSE=True)
import configurations.utility as util
from modelEvaluation.predict import predict, generate_filename
from dataManipulation.dataPreparation import getData
from modelEvaluation.detect import detect_models, detect_latest
from modelEvaluation.calibrate import calibrate


# %%

def compare_cohorts(variable: str, model_name: str, X, y, preds, K=20000, **kwargs):
    options=kwargs.get('options',None)
    if not options:
        options=X[variable].unique()
    preds_with_cohort_variable=pd.merge(preds, X[['PATIENT_ID',variable]],
                                        on='PATIENT_ID').groupby(variable)
    metrics = {'Score': {}, f'Recall_{K}': {}, f'PPV_{K}': {},
               'Brier': {}, 'Brier Before': {}, 'AP': {}}
    for option in options:
        group=preds_with_cohort_variable.get_group(option)
        print( f'{model_name} {variable}={option}')
        obs=np.where(group.OBS >= 1, 1, 0)
        metrics['Score'][f'{model_name}{variable}={option}'] = roc_auc_score(obs, group.PREDCAL)
        metrics[f'Recall_{K}'][f'{model_name}{variable}={option}'], metrics[f'PPV_{K}'][f'{model_name}{variable}={option}'], _, _ = performance(obs,
                                                                              group.PREDCAL, K)
        metrics['Brier'][f'{model_name}{variable}={option}'] = brier_score_loss(obs, group.PREDCAL)
        metrics['Brier Before'][f'{model_name}{variable}={option}'] = brier_score_loss(obs, group.PRED)
        metrics['AP'][f'{model_name}{variable}={option}'] = average_precision_score(obs, group.PREDCAL)

    return metrics
def compare_nested(available_models, X, y, year):
    available_models = [m for m in available_models if ('nested' in m)]
    available_models.sort()
    variable_groups = [r'PATIENT_ID|FEMALE|AGE_[0-9]+$', 'EDC_', 'RXMG_', 'ACG']
    predictors = {}
    for i in range(1, len(variable_groups) + 1):
        predictors[available_models[i - 1]] = r'|'.join(variable_groups[:i])
    metrics = compare(available_models, X, y, year, predictors=predictors)
    return (metrics)


def compare(selected, X, y, year, 
            experiment_name=Path(config.MODELPATH).parts[-1],
            **kwargs):
    import traceback
    K = kwargs.get('K', 20000)
    cohorts=kwargs.get('cohorts', None)
    cohort_metrics= {'Score': {}, f'Recall_{K}': {}, f'PPV_{K}': {},
               'Brier': {}, 'Brier Before': {}, 'AP': {}}
    predictors = kwargs.get('predictors', {m: config.PREDICTORREGEX for m in selected})
    metrics = {'Score': {}, f'Recall_{K}': {}, f'PPV_{K}': {},
               'Brier': {}, 'Brier Before': {}, 'AP': {}}
    for m in selected:
        print(m)
        try:
            probs = calibrate(m, year, experiment_name=experiment_name, presentX=X, presentY=y,
                              predictors=predictors[m], **kwargs)
            if (probs is None):  # If model not found
                continue
            obs = np.where(probs.OBS >= 1, 1, 0)
            metrics['Score'][m] = roc_auc_score(obs, probs.PREDCAL)
            metrics[f'Recall_{K}'][m], metrics[f'PPV_{K}'][m], _, _ = performance(np.where(probs.OBS >= 1, 1, 0),
                                                                                  probs.PREDCAL, K)
            metrics['Brier'][m] = brier_score_loss(obs, probs.PREDCAL)
            metrics['Brier Before'][m] = brier_score_loss(obs, probs.PRED)
            metrics['AP'][m] = average_precision_score(obs, probs.PREDCAL)
            cohort_metrics_model=compare_cohorts(cohorts,m, X, y, probs, K)

            for k,dic in cohort_metrics_model.items():
                for key, value in dic.items():
                    cohort_metrics[k][key]=value
        except Exception as exc:
            print('Something went wrong for model ', m)
            print(traceback.format_exc())
            print(exc)
        if cohorts:
            for k,dic in cohort_metrics.items():
                for key, value in dic.items():
                    metrics[k][key]=value
            # metrics=pd.concat([pd.DataFrame(metrics), pd.DataFrame(cohort_metrics)])
    return (metrics)


import numpy as np
from sklearn.metrics import confusion_matrix


def performance(obs, pred, K, computemetrics=True):
    orderedPred = sorted(pred, reverse=True)
    orderedObs = sorted(obs, reverse=True)
    cutoff = orderedPred[K - 1]
    # print(f'Cutoff value ({K} values): {cutoff}')
    # print(f'Observed cutoff value ({K} values): {orderedObs[K-1]}')
    newpred = pred >= cutoff
    # print('Length of selected list ',sum(newpred))
    if 'COSTE_TOTAL_ANO2' in config.COLUMNS:  # maybe better: not all([int(i)==i for i in obs])
        newobs = obs >= orderedObs[K - 1]
    else:
        newobs = np.where(obs >= 1, 1, 0)  # Whether the patient had ANY admission
    c = confusion_matrix(y_true=newobs, y_pred=newpred)
    print(c)
    tn, fp, fn, tp = c.ravel()
    if not computemetrics:
        return (tn, fp, fn, tp)
    print(' tn, fp, fn, tp =', tn, fp, fn, tp)
    recall = c[1][1] / (c[1][0] + c[1][1])
    ppv = c[1][1] / (c[0][1] + c[1][1])
    specificity = tn / (tn + fp)
    print('Recall, PPV, Spec = ', recall, ppv, specificity)
    return (recall, ppv, specificity, newpred)


def parameter_distribution(models, **args):
    model_dict, grid, params, times_selected = {}, {}, {}, {}
    for m in models:
        print(m)
        try:
            grid[m] = json.load(open(config.USEDCONFIGPATH + config.EXPERIMENT + '/hgb_19.json'))["RANDOM_GRID"]
        except KeyError:
            print('No RANDOM_GRID found in config :(')
        try:
            model_dict[m] = job.load(config.MODELPATH + m + '.joblib')
            params[m] = model_dict[m].get_params()
        except FileNotFoundError:
            print('Model not found :(')
    print(params)
    print(model_dict)
    for parameter, options in grid[m].items():
        times_selected[parameter] = {}
        for m in models:
            opt = params[m][parameter]
            try:
                times_selected[parameter][opt] += 1
            except KeyError:
                times_selected[parameter][opt] = 1
    for parameter in times_selected.keys():
        times_selected[parameter] = pd.DataFrame(times_selected[parameter],
                                                 index=[
                                                     0])  # fixes ValueError: If using all scalar values, you must pass an index
        plt.figure()
        times_selected[parameter].plot(kind='bar', title=parameter, rot=0)



def model_percentiles(df, metric, plot, *args, **kwargs):
    def q1(x):
        return x.quantile(0.05, interpolation='nearest')

    def q2(x):
        return x.quantile(0.5, interpolation='nearest')

    def q3(x):
        return x.quantile(0.95, interpolation='nearest')

    brierdist = df.groupby(['Algorithm'])[metric].agg([q1, q2, q3]).stack(level=0)
    print(f'{metric} distribution per algorithm: ')
    print(brierdist)
    roc, pr = {}, {}
    low, median, high = {}, {}, {}
    models_to_plot = {}
    selected_models = []
    for alg in df.Algorithm.unique():
        df_alg = df.loc[df.Algorithm == alg].to_dict(orient='list')
        perc25 = brierdist.loc[alg]['q1']
        perc50 = brierdist.loc[alg]['q2']
        perc75 = brierdist.loc[alg]['q3']
        low[alg] = list(df_alg['Model'])[list(df_alg[metric]).index(perc25)]
        median[alg] = list(df_alg['Model'])[list(df_alg[metric]).index(perc50)]
        high[alg] = list(df_alg['Model'])[list(df_alg[metric]).index(perc75)]
        # selected_models=[low[alg],median[alg],high[alg]]
        models_to_plot[alg] = {'Perc. 05': low[alg], 'Perc. 50': median[alg], 'Perc. 95': high[alg]}
        roc[alg], pr[alg] = plot(models_to_plot[alg], *args, **kwargs)
    return (roc, pr)


# %%
if __name__ == '__main__':
    year = int(input('YEAR TO PREDICT: ')) if not hasattr(args, 'year') else args.year
    nested = eval(input('NESTED MODEL COMPARISON? (True/False) ')) if not hasattr(args, 'nested') else args.nested
    all_models = eval(input('COMPARE ALL MODELS? (True/False) ')) if not hasattr(args, 'all') else args.all
    cohort_variable=input('Cohort variable:')
    available_models = detect_models()

    if nested:
        selected = sorted([m for m in available_models if ('nested' in m)])
    elif all_models:
        selected = [m for m in available_models if not ('nested' in m)]
    else:
        selected = detect_latest(available_models)

    if Path(config.METRICSPATH +  f'/metrics{year}.csv').is_file():
        available_metrics = pd.read_csv(config.PREDPATH + f'/metrics{year}.csv')
    else:
        available_metrics = pd.DataFrame.from_dict({'Model': []})
    if all([s in available_metrics.Model.values for s in selected]):
        print('All metrics are available')
        print(available_metrics)
        available_metrics['Algorithm'] = [re.sub('_|[0-9]', '', model) for model in available_metrics['Model'].values]
        print(available_metrics.groupby('Algorithm').describe().transpose())

        df = available_metrics
    else:
        selected = [s for s in selected if not (s in available_metrics.Model.values)]

        X, y = getData(year - 1)
        try:
            pastX, pasty = getData(year - 2)
        except AssertionError:
            print(f'getData: HOSPITALIZATION DATA FOR YEAR {year-2} NOT AVAILABLE. PERFORMING INTERNAL VALIDATION.')
            pastX, pasty = X,y
        if not nested:
            metrics = compare(selected, X, y, year, pastX=pastX, pastY=pasty,cohorts=cohort_variable)

        if nested:
            metrics = compare_nested(available_models, X, y, year)
            variable_groups = [r'SEX+ AGE', '+ EDC_', '+ RXMG_', '+ ACG']
            score, recall, ppv, brier, brierBefore, ap = [list(array.values()) for array in list(metrics.values())]
            print(pd.DataFrame(list(zip(selected, variable_groups, score, recall, ppv)),
                               columns=['Model', 'Predictors'] + list(metrics.keys())).to_markdown(index=False))
        else:
            score, recall, ppv, brier, brierBefore, ap = [list(array.values()) for array in list(metrics.values())]
            df = pd.DataFrame(list(zip(metrics['Score'].keys(), score, recall, ppv, brier, brierBefore, ap)),
                              columns=['Model'] + list(metrics.keys()))
            print(df.to_markdown(index=False, ))

        df = pd.concat([df, available_metrics], ignore_index=True, axis=0)
        df['Algorithm'] = [re.sub('_|[0-9]', '', model) for model in df['Model'].values]

        df.to_csv(config.METRICSPATH +  f'/metrics{cohort_variable}{year}.csv', index=False)

                

        print(df.groupby('Algorithm').describe().transpose())

