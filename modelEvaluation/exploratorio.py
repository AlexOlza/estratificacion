#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:25:09 2021

@author: aolza
"""


""" FRAILTY- EDU: Me gustaría saber de aquellos con la etiqueta de fragilidad en 2017
 (serán unos 70000-80000) qué porcentaje de ellos ingresó en 2018 
 (no programado. los criterios que teníamos en Almería). 
 Y si puedes, también qué % de ellos tenía una probabilidad de ingreso >=40% 
 según el modelo que quieras (logística, neuronal…)
 """
from pathlib import Path
import sys
path = str(Path('/home/aolza/Desktop/estratificacion'))
sys.path.insert(0, path)
from configurations import config
from dataManipulation.generarTablasVariables import load

print(dir(config))

import pandas as pd

df=load(config.ACGfiles[2017],predictors=['PATIENT_ID','FRAILTY','ingresoUrg'])
#%%
print('Número de frágiles: ',sum(df.FRAILTY))#en 2017
print('Porcentaje frágiles con ingreso 2018: ',100*len(df.loc[(df.FRAILTY==1) & (df.ingresoUrg>0)])/sum(df.FRAILTY))
print('Porcentaje ingresos 2018 correspondientes a pac. frágiles: ',100*len(df.loc[(df.FRAILTY==1) & (df.ingresoUrg>0)])/len(df.loc[(df.ingresoUrg>0)]))

from tabulate import tabulate
import glob
files=glob.glob(config.rootPath+"predecirIngresos/prediccionesCalibradas/smooth*.csv")
models=[f.split('/')[-1].replace('smooth','').replace('Calibrated','').replace('18.csv','') for f in files if 'AdaBoost' not in f]
porc=[]
ing=[]
for f in files:
    if 'AdaBoost' in f:
        continue
    # print(model)
    pred=pd.read_csv(f)
    aux=pd.merge(df,pred,on='PATIENT_ID')
    porcentaje40=100*len(aux.loc[(aux.FRAILTY==1) & (aux.PROB_INGURG_CAL>0.4)])/len(aux.loc[(aux.FRAILTY==1)])
    porc.append(porcentaje40)
    ing.append(100*len(aux.loc[(aux.FRAILTY==1) & (aux.PROB_INGURG_CAL>0.4) & (aux.INGURG)])/len(aux.loc[(aux.FRAILTY==1) & (aux.PROB_INGURG_CAL>0.4)]))
    # print('Porcentaje frágiles con p>40%: ',porcentaje40)
    # ax=aux.hist(column='PROB_INGURG_CAL',by='FRAILTY')
    # ax.set_title(f.split('/')[-1])
    #%%
table=pd.DataFrame()
table['modelo']=models
table['porcentaje']=porc
table['ingresaron']=ing
print(tabulate(table, headers='keys', tablefmt='psql'))
    