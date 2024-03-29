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

@author: alex
"""
#%% EXTERNAL IMPORTS
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')
import pandas as pd

from scipy.stats import norm
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay

#%% CONFIGURE 

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
try:
    usedconfigpath=os.environ['USEDCONFIG_PATH']
except:
    usedconfigpath=sys.argv[3]
# experiment=input('Experiment: ')
config_used=os.path.join(usedconfigpath,f'{sys.argv[2]}/logisticSexInteraction.json')

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
N=5
if not os.path.exists(filename):
    model=job.load(config.MODELPATH+'logisticSexInteraction.joblib')
    year=2018
    X,_=getData(year-1)
    
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    
    X.drop('PATIENT_ID', axis=1, inplace=True)
    try:
        X.drop('AGE_85GT', axis=1, inplace=True)
    except:
        pass
    
    features=X.columns
    interactions= X.drop([ 'FEMALE'], axis=1).multiply(X.FEMALE,axis=0).astype(np.int8)
    interactions.rename(columns={c:f'{c}INTsex' for c in interactions}, inplace=True)
    X=pd.concat([X,interactions],axis=1)

    stderrInt, zInt, pInt=beta_std_error(model, X)
    print('Intercept: ',list(model.intercept_))
    print('Std. error for the intercept: ',stderrInt[0])
    
    beta=list(model.intercept_)+list(model.coef_[0])
    low, high = confidence_interval_odds_ratio(beta,stderrInt, 0.95)
    
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
    if not 'descripcion' in oddsContrib.columns:
        oddsContrib=translateVariables(oddsContrib) 
        
    year=2018
    X,y=getData(year-1)
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    # X.drop('PATIENT_ID', axis=1, inplace=True)

    interactions= X.drop(['PATIENT_ID', 'FEMALE'], axis=1).multiply(X.FEMALE,axis=0).astype(np.int8)
    interactions.rename(columns={c:f'{c}INTsex' for c in interactions}, inplace=True)
    
    X=pd.concat([X, interactions], axis=1)
    
    Xhom=X.loc[male]
    Xmuj=X.loc[female]
    ymuj=y.loc[y.PATIENT_ID.isin(Xmuj.PATIENT_ID)]
    yhom=y.loc[y.PATIENT_ID.isin(Xhom.PATIENT_ID)]
    
    oddsContrib['NMuj']=[Xmuj[re.sub('INTsex','',name)].sum() for name in oddsContrib.codigo]
    oddsContrib['NHom']=[Xhom[re.sub('INTsex','',name)].sum() for name in oddsContrib.codigo]
    oddsContrib['NMuj_ingreso']=[len(ymuj.loc[y.PATIENT_ID.isin(Xmuj.loc[Xmuj[re.sub('INTsex','',name)]>0].PATIENT_ID)].urgcms.to_numpy().nonzero()[0]) for name in oddsContrib.codigo]
    oddsContrib['NMuj_ingreso']=oddsContrib['NMuj_ingreso']/oddsContrib.NMuj*100
    oddsContrib['NHom_ingreso']=[len(yhom.loc[y.PATIENT_ID.isin(Xhom.loc[Xhom[re.sub('INTsex','',name)]>0].PATIENT_ID)].urgcms.to_numpy().nonzero()[0]) for name in oddsContrib.codigo]
    oddsContrib['NHom_ingreso']=oddsContrib['NHom_ingreso']/oddsContrib.NHom*100
    oddsContrib[['Low', 'Odds', 'High', 'Beta', 'StdErr(beta)', 'codigo', 'NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso', 'descripcion']].to_csv(filename, index=False)

    
assert all(oddsContrib.Low<=oddsContrib.Odds)
assert all(oddsContrib.High>=oddsContrib.Odds)

interactions=oddsContrib.loc[oddsContrib.codigo.str.endswith('INTsex')]
significantRiskWomen=interactions.loc[(interactions.Low>1)]
significantRiskMen=interactions.loc[(interactions.High<1)]
print(f'TOP {N} variables cuya presencia acrecienta el riesgo para las mujeres más que para los hombres: ')
#mayores interacciones positivas
print(significantRiskWomen.sort_values(by='Odds', ascending=False)[['codigo', 'Low','Odds','descripcion','NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso']].head(N).to_markdown(index=False))
print('  ')
significantRiskWomen.sort_values(by='Odds', ascending=False)[[ 'codigo', 'Low','Odds', 'High','descripcion', 'NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso',]].to_csv(os.path.join(config.PREDPATH, f'{config.ALGORITHM}_moreRiskWomen.csv'),
                                                                                                                                     index=False,
                                                                                                                                     sep='\t')

print(f'TOP {N} variables cuya presencia acrecienta el riesgo para los hombres más que para las mujeres: ')
significantRiskMen['invOdds']=1/significantRiskMen.Odds
print(significantRiskMen.sort_values(by='invOdds', ascending=False)[[ 'invOdds','High', 'descripcion', 'NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso',]].to_markdown(index=False))
print('  ')

significantRiskMen.sort_values(by='invOdds', ascending=False)[[ 'codigo', 'Low','Odds', 'invOdds', 'High','descripcion', 'NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso',]].to_csv(os.path.join(config.PREDPATH, f'{config.ALGORITHM}_moreRiskMen.csv'),
                                                                                                                                                 index=False, sep='\t')
#%%
print('TÉRMINOS SIGNIFICATIVOS (NO DE INTERACCIÓN POR SEXO)')
signif_overall=oddsContrib.loc[oddsContrib.Low>1]
print(signif_overall.sort_values(by='Odds', ascending=False)[['Low','Odds','descripcion','NMuj', 'NHom','NMuj_ingreso', 'NHom_ingreso']].head(10).to_markdown(index=False))
