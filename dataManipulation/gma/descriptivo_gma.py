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
X,y=getData(year)
X=X[[c for c in X if X[c].max()>0]]

gmas=X.filter(regex='GMA_*').sum().to_frame()
dic={'':'Sano o pat.agudas',
     '10':'Sano o pat.agudas',
     '20':'Embarazos y partos',
     '31':'pat. crónicas 1 sistema',
     '32': 'pat. crcónicas 2 o 3 sistemas',
     '33': 'pat. crcónicas +3 sistemas',
     '40':'neoplasias'}

gmas['Grupo']=[dic[v.split('_')[-1][:-1]] for v in gmas.index]
#%%
gmas_and_death=pd.merge(X.filter(regex='GMA_*|PATIENT_ID'),y,on='PATIENT_ID').drop('PATIENT_ID',axis=1)
for value, df in gmas_and_death.groupby('DEATH_1YEAR'):
    gmas[f'dead={value}']=df.sum().drop('DEATH_1YEAR',axis=0)
gmas['Muerte %']=gmas['dead=1']*100/gmas[0]
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
    t=df['Muerte %']
    for tt,p in zip(t,ax[i].patches):
        ax[i].annotate(round(tt,2), (p.get_x() * 1.05, p.get_height() * 1.05))

    i+=1

plt.suptitle(f'Número de pacientes con cada GMA en {year}, y porcentaje de muerte')
plt.tight_layout(pad=2)

#%%

