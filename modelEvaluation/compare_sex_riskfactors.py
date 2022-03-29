#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:01:15 2022

@author: aolza
"""
#%% EXTERNAL IMPORTS
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd

from scipy.stats import norm
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay

#%% CONFIGURE 

config_used='/home/aolza/Desktop/estratificacion/configurations/used/noacg_urgcms_excl_nbinj/logisticSexInteraction.json'

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)
#%% INTERNAL IMPORTS
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
from modelEvaluation.independent_sex_riskfactors import beta_std_error, confidence_interval_odds_ratio

#%%
""" QUESTION 2:
    WHICH VARIABLES DIFFER THE MOST WHEN PRESENT IN MEN VS WOMEN?
"""
filename=os.path.join(config.MODELPATH, f'{config.ALGORITHM}_sexSpecificVariables.csv')
if not os.path.exists(filename):
    model=job.load(config.MODELPATH+'logisticSexInteraction.joblib')
    year=2018
    X,_=getData(year-1)
    
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    
    X.drop('PATIENT_ID', axis=1, inplace=True)
    for column in X:
        if column!='FEMALE':
            X[f'{column}INTsex']=X[column]*X['FEMALE']
    features=X.columns
    stderrInt, zInt, pInt=beta_std_error(model, X)
    print('Intercept: ',list(model.intercept_))
    print('Std. error for the intercept: ',stderrInt[0])
    
    beta=list(model.intercept_)+list(model.coef_[0])
    low, high = confidence_interval_odds_ratio(beta,stderrInt, zInt)
    
    #excluding the intercept from now on
    beta={name:value for name, value in zip(features, model.coef_[0])}
    stderr={name:value for name, value in zip(features, stderrInt[1:])}
    
    #Conf.int. for the odds ratio:
    oddsContribLow={name:value for name, value in zip(features, low)}
    oddsContrib={name:np.exp(value) for name, value in zip(features, model.coef_[0])}
    oddsContribHigh={name:value for name, value in zip(features, high)}
    
    oddsContrib={name:[low, odds, high, beta, std] for name, low, odds, high, beta, std in zip(features, 
                                                                       oddsContribLow.values(),
                                                                       oddsContrib.values(),
                                                                       oddsContribHigh.values(),
                                                                       beta.values(),
                                                                       stderr.values())}
    oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',
                                       columns=['Low','Odds','High','Beta', 'StdErr(beta)'])

    oddsContrib.to_csv(filename)
else:
    
    oddsContrib=pd.read_csv(filename)
    from modelEvaluation.independent_sex_riskfactors import translateVariables
    oddsContrib=translateVariables(oddsContrib)
    

# ratioHM=oddsContrib.loc[(oddsContrib['LowratioH/M']>=1) & (oddsContrib['ratioH/M']>=1)]
# print(f'TOP {N} variables cuya presencia acrecienta el riesgo para los hombres más que para las mujeres: ')
# print(oddsContrib.sort_values(by='ratioH/M', ascending=False).head(N)[['ratioH/M','NMuj', 'NHom','descripcion']])
# print('  ')

# ratioMH=oddsContrib.loc[(oddsContrib['LowratioM/H']>=1) & (oddsContrib['ratioM/H']>=1)]
# print(f'TOP {N} variables cuya presencia acrecienta el riesgo para las mujeres más que para los hombres: ')
# print(oddsContrib.sort_values(by='ratioM/H', ascending=False).head(N)[['ratioM/H','NMuj', 'NHom','descripcion']])
