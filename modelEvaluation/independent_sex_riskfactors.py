#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:52:00 2022

@author: aolza
"""
from scipy.stats import norm
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd
#%%
from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    config_used=input('Full path to configuration json file...: ')
    configuration=util.configure(config_used)
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
#%%
def translateVariables(df,**kwargs):
     dictionaryFile=kwargs.get('file',os.path.join(config.INDISPENSABLEDATAPATH+'diccionarioACG.csv'))
     dictionary=pd.read_csv(dictionaryFile)
     #Add interaction column codes if present:
     for column in df.codigo.values:
        if column.endswith('INTsex'):
            dictionary=dictionary.append(pd.DataFrame([[column,dictionary.loc[dictionary.codigo==re.sub('INTsex','',column)].descripcion.values[0]]],
                                           columns=['codigo','descripcion']),
                                            ignore_index=True)
     df=pd.merge(df, dictionary, on='codigo')
     return df
 
def beta_std_error(logModel, X, eps=1e-20):
    """ Source:
        https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
    """
    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = logModel.predict_proba(X)
    
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])

    # Diagonal matrix with each predicted observation's p(1-p)
    V = np.product(predProbs, axis=1) #avoids memory issues using shape (n,) instead of (n,n)
    
    # Covariance matrix
    # * does ordinary multiplication. We can use it because V is diagonal :)
    MM=X_design.T * V #shape pxn
    # Note that the @-operator does matrix multiplication in Python 3.5+
    M= MM@ X_design
    covLogit = np.linalg.pinv(M) #avoids singularity issues using Moore-Penrose pseudoinverse 
    del M, MM, V, X_design
    # Standard errors
    D=np.diag(covLogit).copy() #copy, because the original array is non-mutable (ValueError: assignment destination is read-only)
    D[np.abs(D) < eps] = 0 # Avoid negative values in the diagonal (a necessary evil)
    
    assert all(D>=0), 'Negative values found in the diagonal of the var-cov matrix. Increase eps!!'
    
    stderr=np.sqrt(D)
    print("Standard errors: ", stderr)
    # Wald statistic (coefficient / s.e.) ^ 2
    logitParams = np.insert(logModel.coef_, 0, logModel.intercept_)
    waldT=(logitParams / stderr) ** 2
    print("Wald statistics: ", waldT)
    p = (1 - norm.cdf(abs(waldT))) * 2
    return( stderr, waldT, p)

def confidence_interval_odds_ratio(betas, stderr, waldT):
    """ Source:
        https://stats.stackexchange.com/questions/354098/calculating-confidence-intervals-for-a-logistic-regression
     Using the invariance property of the MLE allows us to exponentiate to get the conf.int.
    """
    low=np.exp(betas-waldT*stderr)
    high=np.exp(betas+waldT*stderr)
    return(low,high)

if __name__=="__main__":
    year=int(input('YEAR TO PREDICT: ')) 
    X,y=getData(year-1)
    #%%
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    
    X.drop(['FEMALE', 'PATIENT_ID'], axis=1, inplace=True)
    #%%
    K=20000
    #%%
    separateFeatures=X.columns
    
    modeloH=job.load(config.MODELPATH+'logisticHombres.joblib')
    modeloM=job.load(config.MODELPATH+'logisticMujeres.joblib')
    
    stderrH, zH, pH=beta_std_error(modeloH, X)
    stderrM, zM, pM=beta_std_error(modeloM, X)
    
    stderrHdict={name:value  for name, value in zip(separateFeatures, stderrH)}
    stderrMdict={name:value  for name, value in zip(separateFeatures, stderrH)}
    
    betaH=list(modeloH.intercept_)+list(modeloH.coef_[0])
    betaM=list(modeloM.intercept_)+list(modeloM.coef_[0])
    lowH, highH = confidence_interval_odds_ratio(betaH,stderrH, zH)
    lowM, highM = confidence_interval_odds_ratio(betaM,stderrM, zM)
    
    oddsContribMujLow={name:value for name, value in zip(separateFeatures, lowM)}
    oddsContribMuj={name:np.exp(value) for name, value in zip(separateFeatures, modeloM.coef_[0])}
    oddsContribMujHigh={name:value for name, value in zip(separateFeatures, highM)}
    
    oddsContribHomLow={name:value for name, value in zip(separateFeatures, lowH)}
    oddsContribHom={name:np.exp(value) for name, value in zip(separateFeatures, modeloH.coef_[0])}
    oddsContribHomHigh={name:value for name, value in zip(separateFeatures, highH)}
    
    oddsContrib={name:[lowm, muj, highm, lowh, hom, highh, stdM, stdH] for name, lowm, muj, highm, lowh, hom, highh, stdM, stdH in zip(separateFeatures, 
                                                                       oddsContribMujLow.values(),
                                                                       oddsContribMuj.values(),
                                                                       oddsContribMujHigh.values(),
                                                                       oddsContribHomLow.values(),
                                                                       oddsContribHom.values(),
                                                                       oddsContribHomHigh.values(),
                                                                       stderrMdict.values(),
                                                                       stderrHdict.values())}
    oddsContrib=pd.DataFrame.from_dict(oddsContrib,orient='index',
                                       columns=['LowM','Mujeres','HighM','LowH', 'Hombres', 'HighH', 'stderrM', 'stderrH'])
    
    oddsContrib['LowratioH/M']=oddsContrib['LowH']/oddsContrib['LowM']
    oddsContrib['ratioH/M']=oddsContrib['Hombres']/oddsContrib['Mujeres']
    oddsContrib['HighratioH/M']=oddsContrib['HighH']/oddsContrib['HighM']
    oddsContrib['LowratioM/H']=1/oddsContrib['LowratioH/M']
    oddsContrib['ratioM/H']=1/oddsContrib['ratioH/M']
    oddsContrib['HighratioM/H']=1/oddsContrib['HighratioH/M']
    Xhom=X.loc[male]
    Xmuj=X.loc[female]
    oddsContrib['NMuj']=[Xmuj[name].sum() for name in oddsContrib.index]
    oddsContrib['NHom']=[Xhom[name].sum() for name in oddsContrib.index]
    oddsContrib['codigo']=oddsContrib.index
    oddsContrib=translateVariables(oddsContrib)
    
    """ QUESTION 1:
        WHAT ARE THE STRONGEST RISK FACTORS FOR WOMEN AND MEN SEPARATELY?
    """
    #%% PRINT RISK FACTORS
    N=5
    for s, col, low in zip(['Mujeres', 'Hombres'], ['NMuj', 'NHom'], ['LowM','LowH']):
        print(f'TOP {N} factores de riesgo para {s}')
        print(oddsContrib.sort_values(by=s, ascending=False).head(N)[[s,low,col,'descripcion']])
        print('  ')
        print(f'TOP {N} factores protectores para {s}')
        print(oddsContrib.sort_values(by=s, ascending=True).head(N)[[s, low,col,'descripcion']])
        print('  ')
    print('-----'*N)
    print('  ')
    
    #%%
    oddsContrib.to_csv(config.MODELPATH+'sexSpecificOddsContributions.csv')
    oddsContrib=pd.read_csv(config.MODELPATH+'sexSpecificOddsContributions.csv')
    print('FACTORES DE RIESGO SIGNIFICATIVOS EN MUJERES: ')
    riskFactors=oddsContrib.loc[(oddsContrib['LowM']>=1) & (oddsContrib.Mujeres>=1)]
    print(riskFactors.sort_values(by='Mujeres', ascending=False)[['codigo','Mujeres', 'NMuj','descripcion']])
    
    print('FACTORES DE RIESGO SIGNIFICATIVOS EN HOMBRES: ')
    riskFactors=oddsContrib.loc[(oddsContrib['LowH']>=1) & (oddsContrib.Mujeres>=1)]
    print(riskFactors.sort_values(by='Hombres', ascending=False)[['codigo','Hombres', 'NHom','descripcion']])
