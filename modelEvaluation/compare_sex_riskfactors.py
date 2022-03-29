#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:01:15 2022

@author: aolza
"""
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
from modelEvaluation.predict import generate_filename
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd
#%%

config_used=input('Full path to configuration json file ')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)
from modelEvaluation.compare import detect_models
from dataManipulation.dataPreparation import getData
#%%
def translateVariables(df,**kwargs):
     dictionaryFile=kwargs.get('file',os.path.join(config.INDISPENSABLEDATAPATH+'diccionarioACG.csv'))
     dictionary=pd.read_csv(dictionaryFile)
     dictionary.index=dictionary.codigo
     df=pd.merge(df, dictionary, right_index=True, left_index=True)
     return df
 
def beta_std_error(logModel, X):
    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = logModel.predict_proba(X)
    
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    V = np.diagflat(np.product(predProbs, axis=1))
    
    # Covariance matrix
    # Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
    # Python 3.5+, you can replace the covLogit-line below with the more readable:
    covLogit = np.linalg.inv(X_design.T @ V @ X_design)
    # Standard errors
    stderr=np.sqrt(np.diag(covLogit))
    print("Standard errors: ", stderr)
    # Wald statistic (coefficient / s.e.) ^ 2
    logitParams = np.insert(logModel.coef_, 0, logModel.intercept_)
    print("Wald statistics: ", (logitParams / np.sqrt(np.diag(covLogit))) ** 2)
    return stderr
#%%
year=int(input('YEAR TO PREDICT: ')) 
X,y=getData(year-1)
#%%
female=X['FEMALE']==1
male=X['FEMALE']==0
sex=['Mujeres', 'Hombres']

X.drop(['FEMALE', 'PATIENT_ID'], axis=1, inplace=True)
#%%
available_models=detect_models()
K=20000
#%%
separateFeatures=X.columns

modeloH=job.load(config.MODELPATH+'logisticHombres.joblib')
modeloM=job.load(config.MODELPATH+'logisticMujeres.joblib')

stderrH=beta_std_error(modeloH, X)
stderrM=beta_std_error(modeloM, X)

stderrHdict={name:value  for name, value in zip(separateFeatures, stderrH)}
stderrMdict={name:value  for name, value in zip(separateFeatures, stderrH)}

oddsContribMuj={name:np.exp(value) for name, value in zip(separateFeatures, modeloM.coef_[0])}
oddsContribHom={name:np.exp(value) for name, value in zip(separateFeatures, modeloH.coef_[0])}
oddsContrib={name:[muj, hom] for name, muj, hom, stdM, stdH in zip(separateFeatures, 
                                                                   oddsContribMuj.values(), 
                                                                   oddsContribHom.values(),
                                                                   stderrMdict.values(),
                                                                   stderrHdict.values())}
oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',
                                   columns=['Mujeres', 'Hombres', 'stderrM', 'stderrH'])

oddsContrib['ratioH/M']=oddsContrib['Hombres']/oddsContrib['Mujeres']
oddsContrib['ratioM/H']=1/oddsContrib['ratioH/M']
Xhom=X.loc[male]
Xmuj=X.loc[female]
oddsContrib['NMuj']=[Xmuj[name].sum() for name in oddsContrib.index]
oddsContrib['NHom']=[Xhom[name].sum() for name in oddsContrib.index]

oddsContrib=translateVariables(oddsContrib)

#%% PRINT RISK FACTORS
N=5
for s, col in zip(['Mujeres', 'Hombres'], ['NMuj', 'NHom']):
    print(f'TOP {N} factores de riesgo para {s}')
    print(oddsContrib.sort_values(by=s, ascending=False).head(N)[[s,col,'descripcion']])
    print('  ')
    print(f'TOP {N} factores protectores para {s}')
    print(oddsContrib.sort_values(by=s, ascending=True).head(N)[[s, col,'descripcion']])
    print('  ')
print('-----'*N)
print('  ')

print(f'TOP {N} variables cuya presencia acrecienta el riesgo para los hombres más que para las mujeres: ')
print(oddsContrib.sort_values(by='ratioH/M', ascending=False).head(N)[['ratioH/M','NMuj', 'NHom','descripcion']])
print('  ')
print(f'TOP {N} variables cuya presencia acrecienta el riesgo para las mujeres más que para los hombres: ')
print(oddsContrib.sort_values(by='ratioM/H', ascending=False).head(N)[['ratioM/H','NMuj', 'NHom','descripcion']])

#%%
# oddsContrib.to_csv(config.MODELPATH+'sexSpecificOddsContributions.csv')

