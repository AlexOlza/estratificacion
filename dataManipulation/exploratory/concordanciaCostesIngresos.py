#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:38:28 2021

@author: aolza
"""

import os
import numpy as np
import pandas as pd
#%%
filename = "estratificacionDatos/ing2016-2017Activos.csv"
# filename = "estratificacionDatos/ing2017-2018Activos.csv"
# filename = "estratificacionDatos/ing2016-2017Activos.csv"

if not os.path.exists('articulo'):
    os.mkdir('articulo')
#%%  

datos = pd.DataFrame()
for chunk in pd.read_csv(filename, chunksize=100000):
        d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
        d['PATIENT_ID']=np.int64
        chunk= chunk.astype(d)
        datos = pd.concat([datos, chunk], ignore_index=True)


try:    
    datos=datos.drop(labels=['EDC_NUR11','EDC_RES10','RXMG_ZZZX000',
                   'ACG_5320','ACG_5330','ACG_5340'],axis=1)
except KeyError:
    pass

#%%
datos['algunIngresoUrg']=False
datos.iloc[ (datos.loc[datos['ingresoUrg']>0].index), datos.columns.get_loc('algunIngresoUrg')]=True
#%%
"""
Two-sample t-tests:
    Compare the means of two groups under the assumption that
    both samples are random, independent, 
    and normally distributed with unknown but equal variances
"""

import seaborn as sns
sns.boxplot(x='algunIngresoUrg', y='COSTE_TOTAL_ANO2', data=datos,showfliers = False).set_title('2017')
#%%
from scipy.stats import ttest_ind
"""WELCH T_ TEST, equal variance=False"""
ingresados=datos[datos['algunIngresoUrg']==True]['COSTE_TOTAL_ANO2']
otros=datos[datos['algunIngresoUrg']==False]['COSTE_TOTAL_ANO2']
print(ttest_ind(ingresados,otros, equal_var=False))
#%%
ingresados.describe()
otros.describe()
sum(ingresados==0)
#%%
datos.sort_values(by='COSTE_TOTAL_ANO2',ascending=False,inplace=True)
listado=datos.COSTE_TOTAL_ANO2[:22045].index
datos.algunIngresoUrg[listado].describe()
