#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:13:19 2021

@author: alex


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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, RocCurveDisplay, roc_curve, auc, \
    precision_recall_curve, PrecisionRecallDisplay
import sys

sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
import configurations.utility as util
from modelEvaluation.predict import predict, generate_filename
from dataManipulation.dataPreparation import getData
from modelEvaluation.detect import detect_models, detect_latest
from modelEvaluation.calibrate import calibrate


# %%
import numpy as np
from sklearn.metrics import confusion_matrix


def performance(obs, pred, K, computemetrics=True, verbose=True):
    orderedPred = sorted(pred, reverse=True)
    orderedObs = sorted(obs, reverse=True)
    cutoff = orderedPred[K - 1]
    # print(f'Cutoff value ({K} values): {cutoff}')
    # print(f'Observed cutoff value ({K} values): {orderedObs[K-1]}')
    newpred = pred >= cutoff
    # print('Length of selected list ',sum(newpred))
    if  not all([int(i)==i for i in obs]):
        newobs = obs >= orderedObs[K - 1]
    else:
        newobs = np.where(obs >= 1, 1, 0)  # Whether the patient had ANY admission
    c = confusion_matrix(y_true=newobs, y_pred=newpred)
    if verbose: print(c)
    tn, fp, fn, tp = c.ravel()
    if not computemetrics:
        return (tn, fp, fn, tp)
    if verbose: print(' tn, fp, fn, tp =', tn, fp, fn, tp)
    recall = c[1][1] / (c[1][0] + c[1][1])
    ppv = c[1][1] / (c[0][1] + c[1][1])
    specificity = tn / (tn + fp)
    if verbose: print('Recall, PPV, Spec = ', recall, ppv, specificity)
    return (recall, ppv, specificity, newpred)

# %%
if __name__ == '__main__':
    year = 2017
    K=20000
    
    available_models = detect_models()
    
    X,y=getData(year)
    
    deathModel='logistic20221104_132237'
    predDeathCCS, score=predict(deathModel,config.EXPERIMENT,2018,X=X,y=y)

    predCostACG=pd.read_csv('/home/alex/Desktop/estratificacion/predictions/cost_ACG/linear20221205_132708__2018.csv')
    predCostCCS=pd.read_csv('/home/alex/Desktop/estratificacion/predictions/costCCS_pharma/linear20221205_131359__2018.csv')
    
    
    for preds in [predDeathCCS,predCostCCS,predCostACG]:
        preds=pd.merge(preds,y,on='PATIENT_ID')
        performance(preds.DEATH_1YEAR, preds.PRED, K)
        