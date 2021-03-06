#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Identify which variables have the most different effect
    for men and women

PROCEDURE:
    Those variables have the biggest positive and negative
    coefficients for the interaction term with sex.
    The exponential of each coefficient (expb) means that
    the odds of hospitalization for women with that disease
    is expb times higher
    than that of a man with the same disease.

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
experiment=input('Experiment: ')
config_used=os.path.join(os.environ['USEDCONFIG_PATH'],f'{experiment}/logisticSexInteraction.json')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)
#%% INTERNAL IMPORTS
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
from modelEvaluation.independent_sex_riskfactors import beta_std_error, confidence_interval_odds_ratio


#%%
"""
    WHICH VARIABLES DIFFER THE MOST WHEN PRESENT IN MEN VS WOMEN?
"""
filename=os.path.join(config.PREDPATH, f'{config.ALGORITHM}_sexSpecificVariables.csv')
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
    oddsContribLow={name:value for name, value in zip(features, low[1:])}
    oddsContrib={name:np.exp(value) for name, value in zip(features, model.coef_[0])}
    oddsContribHigh={name:value for name, value in zip(features, high[1:])}
    
    oddsContrib={name:[low, odds, high, beta, std] for name, low, odds, high, beta, std in zip(features, 
                                                                       oddsContribLow.values(),
                                                                       oddsContrib.values(),
                                                                       oddsContribHigh.values(),
                                                                       beta.values(),
                                                                       stderr.values())}
    oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',
                                       columns=['Low','Odds','High','Beta', 'StdErr(beta)'])

    oddsContrib['codigo']=oddsContrib.index
    oddsContrib.to_csv(filename, index=False)
else:
    
    oddsContrib=pd.read_csv(filename,sep=',')
    from modelEvaluation.independent_sex_riskfactors import translateVariables
    oddsContrib=translateVariables(oddsContrib)
    N=5
    year=2018
    X,y=getData(year-1)
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    # X.drop('PATIENT_ID', axis=1, inplace=True)
    for column in X:
        if column!='FEMALE':
            X[f'{column}INTsex']=X[column]*X['FEMALE']
    Xhom=X.loc[male]
    Xmuj=X.loc[female]
    
    oddsContrib['NMuj']=[Xmuj[re.sub('INTsex','',name)].sum() for name in oddsContrib.codigo]
    oddsContrib['NHom']=[Xhom[re.sub('INTsex','',name)].sum() for name in oddsContrib.codigo]
    oddsContrib.to_csv(filename, index=False)
interactions=oddsContrib.loc[oddsContrib.codigo.str.endswith('INTsex')]
significantRiskWomen=interactions.loc[(interactions.Low>=1) & (interactions.Odds>=1)]
significantRiskMen=interactions.loc[(interactions.High<=1) & (interactions.Odds<=1)]
print(f'TOP {N} variables cuya presencia acrecienta el riesgo para las mujeres m??s que para los hombres: ')
#mayores interacciones positivas
print(significantRiskWomen.sort_values(by='Odds', ascending=False)[[ 'codigo','Odds','descripcion','NMuj', 'NHom']])
print('  ')

significantRiskWomen.sort_values(by='Odds', ascending=False)[[ 'codigo', 'Low','Odds', 'High','descripcion', 'NMuj', 'NHom']].to_csv(os.path.join(config.PREDPATH,'masRiesgoMujeres.csv'), index=False)
print(f'TOP {N} variables cuya presencia acrecienta el riesgo para los hombres m??s que para las mujeres: ')
significantRiskMen['invOdds']=1/significantRiskMen.Odds
print(significantRiskMen.sort_values(by='invOdds', ascending=False)[[ 'invOdds', 'descripcion', 'NMuj', 'NHom']])
print('  ')

significantRiskMen.sort_values(by='invOdds', ascending=False)[[ 'codigo', 'Low','Odds', 'invOdds', 'High','descripcion', 'NMuj', 'NHom']].to_csv(os.path.join(config.PREDPATH,'masRiesgoHombres.csv'), index=False)