#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:39:29 2022

@author: aolza
"""
#%% Descriptivo GMA
import pandas as pd
import os
import importlib

chosen_config='configurations.cluster.logistic'
importlib.invalidate_caches()
from python_settings import settings as config
settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(settings) # configure() receives a python module
assert config.configured 

from dataManipulation.dataPreparation import getData
#%%
year=eval(input('Year: '))
X,y=getData(year,columns=['DEATH_1YEAR'],
            additional_columns=['GMA_peso-ip'])
_, coste=getData(year,columns=['COSTE_TOTAL_ANO2'])
_, ingreso=getData(year,columns=['urgcms'])
X=X[[c for c in X if X[c].max()>0]]

import numpy as np
ingreso.urgcms=np.where(ingreso.urgcms>=1,1,0)
respuestas=pd.merge(pd.merge(y,coste,on='PATIENT_ID'),ingreso,on='PATIENT_ID')

gmas=X.filter(regex='GMA_[0-9]+').sum().to_frame()
dic={'':'Sano o pat.agudas',
     '10':'Sano o pat.agudas',
     '20':'Embarazos y partos',
     '31':'pat. crónicas 1 sistema',
     '32': 'pat. crcónicas 2 o 3 sistemas',
     '33': 'pat. crcónicas +3 sistemas',
     '40':'neoplasias'}

gmas['Grupo']=[dic[v.split('_')[-1][:-1]] for v in gmas.index]

#%%
gmas_and_death=pd.merge(X.filter(regex='GMA_[0-9]+|PATIENT_ID'),respuestas,on='PATIENT_ID')
for value, df in gmas_and_death.groupby('DEATH_1YEAR'):
    gmas[f'dead={value}']=df.sum().drop('DEATH_1YEAR',axis=0)
gmas['Muerte %']=gmas['dead=1']*100/gmas[0]

for value, df in gmas_and_death.groupby('urgcms'):
    gmas[f'hosp={value}']=df.sum().drop('urgcms',axis=0)
gmas['Ing %']=gmas['hosp=1']*100/gmas[0]

gmas_and_death['GMA']=gmas_and_death.filter(regex='GMA_[0-9]+').idxmax(axis=1)

for value, df in gmas_and_death.groupby('GMA'):
    gmas.loc[value,'coste_medio']=df.COSTE_TOTAL_ANO2.mean()

gmas['count']=gmas[0]
#%%
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
i=0
fig, axes=plt.subplots(3,2,figsize=(15,15), sharey=True)
ax=axes.ravel()
labels=['Sano o pat.agudas', 'Embarazos y partos',
       'pat. crón. 1 sist.', 'pat. crón. 2 o 3 sist.',
       'pat. cron. +3 sist.', 'neoplasias']
grupos=['Sano o pat.agudas', 'Embarazos y partos',
       'pat. crónicas 1 sistema', 'pat. crcónicas 2 o 3 sistemas',
       'pat. crcónicas +3 sistemas', 'neoplasias']
for grupo, label in zip(grupos,labels):
    df=gmas.loc[gmas.Grupo==grupo]
    df[0].plot.bar(title=label, ax=ax[i],legend=False,sort_columns=True, use_index=False)
    m=df['Muerte %']
    ing=df['Ing %']
    c=df['coste_medio']
    for mm,ii,cc,p in zip(m,ing,c,ax[i].patches):
        ax[i].annotate(round(mm,2), (p.get_x() * 1.05, p.get_height() * 1.05))
        ax[i].annotate(round(ii,2), (p.get_x() * 1.05 , p.get_height() * 2.05))
        ax[i].annotate(round(cc,2), (p.get_x() * 1.05, p.get_height() * 3.05 ))

    i+=1

plt.suptitle(f'Número de pacientes con cada GMA en {year}, y porcentaje de muerte')
plt.tight_layout(pad=2)

#%%
#%%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
i=0
fig, axes=plt.subplots(3,2,figsize=(15,15), sharey=True)
ax=axes.ravel()
labels=['Sano o pat.agudas', 'Embarazos y partos',
       'pat. crón. 1 sist.', 'pat. crón. 2 o 3 sist.',
       'pat. cron. +3 sist.', 'neoplasias']
grupos=['Sano o pat.agudas', 'Embarazos y partos',
       'pat. crónicas 1 sistema', 'pat. crcónicas 2 o 3 sistemas',
       'pat. crcónicas +3 sistemas', 'neoplasias']
for grupo, label in zip(grupos,labels):
    df=gmas.loc[gmas.Grupo==grupo]
    df[['coste_medio']].plot(style='.-',title=label, ax=ax[i],sort_columns=True,use_index=False)
    i+=1
    
plt.suptitle(f'Coste medio de los pacientes con cada GMA en {year}')
plt.tight_layout(pad=2)

#%%
i=0
fig, axes=plt.subplots(3,2,figsize=(15,15))
ax=axes.ravel()
for grupo, label in zip(grupos,labels):
    df=gmas.loc[gmas.Grupo==grupo]
    df[['Muerte %']].plot(style='.-',title=label, ax=ax[i],sort_columns=True,use_index=False)
    df[['Ing %']].plot(style='.-',title=label, ax=ax[i],sort_columns=True,use_index=False)
    i+=1
    
plt.suptitle(f'Ingresos y muerte de los pacientes con cada GMA en {year}')
plt.tight_layout(pad=2)
#%%
print(gmas[['count','coste_medio','Muerte %', 'Ing %']].round(2).to_markdown())

#%%
"""CATEGORIZE peso-ip"""

gmas_and_death=pd.merge(gmas_and_death,X[['PATIENT_ID','GMA_peso-ip']],on='PATIENT_ID')

quantiles=gmas_and_death['GMA_peso-ip'].quantile([0.5,0.8,0.95,0.99])

gmas_and_death['complejidad']=np.where(gmas_and_death['GMA_peso-ip']>=quantiles[0.50],2,1)
gmas_and_death['complejidad']=np.where(gmas_and_death['GMA_peso-ip']>=quantiles[0.80],3,gmas_and_death['complejidad'])
gmas_and_death['complejidad']=np.where(gmas_and_death['GMA_peso-ip']>=quantiles[0.95],4,gmas_and_death['complejidad'])
gmas_and_death['complejidad']=np.where(gmas_and_death['GMA_peso-ip']>=quantiles[0.99],5,gmas_and_death['complejidad'])
#%%

gmas_and_death['complejidad_GMA']=gmas_and_death['GMA'].str.slice(-1)
print('GMA complejidad puntos de corte')
print(gmas_and_death.groupby('complejidad_GMA')['GMA_peso-ip'].aggregate('max'))