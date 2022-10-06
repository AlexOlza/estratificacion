#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:12:12 2022

@author: aolza
"""
from scipy.stats import norm
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd

#%%
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
try:
    usedconfigpath=os.environ['USEDCONFIG_PATH']
except:
    usedconfigpath=sys.argv[3]
# experiment=input('Experiment: ')
config_used=os.path.join(usedconfigpath,f'{sys.argv[2]}/linearMujeres.json')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
from modelEvaluation.independent_sex_riskfactors import translateVariables
from modelEvaluation.regression.sex_functions import beta_std_error,confidence_interval_betas
year=2018#int(input('YEAR TO PREDICT: '))
filename=config.PREDPATH+'/sexSpecificBetas.csv'
#%%
if not Path(filename).is_file():    
    X,y=getData(year-1)
   
    female=X['FEMALE']==1
    male=X['FEMALE']==0
    sex=['Mujeres', 'Hombres']
    Xhom=X.loc[male]
    Xmuj=X.loc[female]
    ymuj=y.loc[y.PATIENT_ID.isin(Xmuj.PATIENT_ID)]
    yhom=y.loc[y.PATIENT_ID.isin(Xhom.PATIENT_ID)]
    X.drop(['FEMALE', 'PATIENT_ID'], axis=1, inplace=True)
    
    K=20000
   
    separateFeatures=X.columns
    fullContrib=pd.DataFrame()
    
    for sexname, Xx, yy in zip(sex, [Xmuj, Xhom],[ymuj,yhom]):
        sexname_abbr=sexname[:3]
        model=job.load(config.MODELPATH+f'linear{sexname}.joblib') 
        betas=list(model.intercept_)+list(model.coef_[0])
        stderr=beta_std_error(model, Xx.drop(['PATIENT_ID','FEMALE'],axis=1), yy[config.COLUMNS])
        low, high=confidence_interval_betas(betas, stderr, confidence_level=0.95)
        
        stderrDict={name:value  for name, value in zip(separateFeatures, stderr)}
      
        
        sexContribLow={name:value for name, value in zip(separateFeatures, low[1:])}
        sexContrib={name:value for name, value in zip(separateFeatures, model.coef_[0])}
        sexContribHigh={name:value for name, value in zip(separateFeatures, high[1:])}
        
        
        sexContrib={name:[L, beta, H, std] for name, L, beta, H, std in zip(separateFeatures, 
                                                                           sexContribLow.values(),
                                                                           sexContrib.values(),
                                                                           sexContribHigh.values(),
                                                                           stderrDict.values())}
        sexContrib=pd.DataFrame.from_dict(sexContrib,orient='index',
                                           columns=[f'Low{sexname_abbr}',f'beta{sexname_abbr}',f'High{sexname_abbr}', f'stderr{sexname_abbr}'])
        
        assert not any(sexContrib[f'beta{sexname_abbr}'].isna())
        assert all(sexContrib[f'Low{sexname_abbr}'].fillna(sexContrib[f'beta{sexname_abbr}'].min())<=sexContrib[f'beta{sexname_abbr}'])
        assert all(sexContrib[f'High{sexname_abbr}'].fillna(sexContrib[f'beta{sexname_abbr}'].max())>=sexContrib[f'beta{sexname_abbr}'])
        

        sexContrib[f'N{sexname_abbr}']=[Xx[name].sum() for name in sexContrib.index]
        # sexContrib[f'N{sexname_abbr}_ingreso']=[len(ymuj.loc[y.PATIENT_ID.isin(Xmuj.loc[Xmuj[re.sub('INTsex','',name)]>0].PATIENT_ID)].urgcms.to_numpy().nonzero()[0]) for name in sexContrib.codigo]
        # sexContrib[f'N{sexname_abbr}_ingreso']=sexContrib[f'N{sexname_abbr}_ingreso']/sexContrib.NMuj*100
        
        fullContrib=pd.concat([fullContrib,sexContrib],axis=1)
        fullContrib['codigo']=sexContrib.index
        
    fullContrib=translateVariables(fullContrib)
    fullContrib.to_csv(filename, index=False)
   
else:
    fullContrib=pd.read_csv(filename)
    
#%%
print('FACTORES DE RIESGO SIGNIFICATIVOS EN MUJERES: ')
riskFactors=fullContrib.loc[(fullContrib['LowMuj']>=0)]
df1=riskFactors.sort_values(by='betaMuj', ascending=False)
print(df1[['codigo','LowMuj','betaMuj',  'NMuj','descripcion']].head(10).to_markdown(index=False))
df1.to_csv(config.PREDPATH+'/significativos_Mujeres.csv', index=False)
#%%
print('FACTORES DE RIESGO SIGNIFICATIVOS EN HOMBRES: ')
riskFactors=fullContrib.loc[(fullContrib['LowHom']>=0) ]
df2=riskFactors.sort_values(by='betaHom', ascending=False)
print(df2[['codigo','LowHom','betaHom', 'NHom', 'descripcion']].head(10).to_markdown(index=False))
df2.to_csv(config.PREDPATH+'/significativos_Hombres.csv', index=False)

#%%
"""
FACTORES MÁS FUERTES, AUNQUE NO SEAN SIGNIFICATIVOS:
"""
N=5
for s, col, low, high in zip(['betaMuj', 'betaHom'], ['NMuj', 'NHom'], 
                       ['LowMuj','LowHom'],['HighMuj','HighHom']):
    print(f'TOP {N} factores de riesgo para {s}')
    print(fullContrib.sort_values(by=s, ascending=False).head(N)[['codigo',s,low,col,'descripcion']].to_markdown(index=False))
    print('  ')
    print(f'TOP {N} factores protectores para {s}')
    print(fullContrib.sort_values(by=s, ascending=True).head(N)[['codigo',s, high,col,'descripcion']].to_markdown(index=False))
    print('  ')
print('-----'*N)
print('  ')
