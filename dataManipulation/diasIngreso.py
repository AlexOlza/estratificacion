#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CÁLCULO DE LOS DÍAS DE INGRESO- TEMPORALMENTE ABANDONADO
Created on Mon Nov 15 12:11:01 2021

@author: aolza
"""
"""
------------------------------------

Continuación de exploratorio/exploratorio.r

#1) 3 TABLAS CON LAS VARIABLES RESPUESTA (2016,2017,2018), 51 COLUMNAS
#política: desde python leeremos sólo las necesarias para cada modelo

#política: Cálculo de los días de ingreso
#si el alta es antes del 31Dic del mismo año: fecalt-fecing,sin_alta=0
#si no, 31Dic-fecing,sin_alta=1
#si el alta es NA, dos opciones:
#       A) Asumir que sigue ingresado: 31Dic-fecing,sin_alta=1
#       B) Imputar: la mediana en el centro y teniendo en cuenta el tipo de ingreso,sin_alta=?


Created on Fri Nov 12 12:01:39 2021

@author: aolza
"""
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import os
f='ingresos2016_2018.csv'
for path in Path('/home/aolza/Desktop/estratificacion').rglob(f):
    ing=pd.read_csv(path)
    break 

ing[['fecing','fecalt']]=ing[['fecing','fecalt']].apply(pd.to_datetime, errors='coerce')#FORMAT

#DIAS DE INGRESO
ing['diasepisodio']=ing.fecalt-ing.fecing

dic16=pd.to_datetime('2016-12-31')
dic17=pd.to_datetime('2017-12-31')
dic18=pd.to_datetime('2018-12-31')
x=pd.to_datetime('2000-01-01') #fecha arbitraria

interanual=(ing.fecing.dt.year!=ing.fecalt.dt.year)
ing['Normalized'] = np.where(((interanual) & (ing.fecing.dt.year==2016)), dic16,x)
ing['Normalized'] = np.where(((interanual) & (ing.fecing.dt.year==2017)), dic17,ing.Normalized)
ing['Normalized'] = np.where(((interanual) & (ing.fecing.dt.year==2018)),  dic18,ing['Normalized'])
ing.Normalized=ing.Normalized.apply(pd.to_datetime, errors='coerce')

ing.diasepisodio=np.where(interanual,ing.Normalized-ing.fecing,ing.diasepisodio)
i=ing.loc[ing.fecing.dt.year!=ing.fecalt.dt.year]
print(i[['fecing','fecalt','Normalized','diasepisodio']])

#todos los ingresos sin alta son de centros psiquiátricos 
#pienso que siguen ingresados
ing[ing.fecalt.isnull()][['fecing','fecalt','Normalized','diasepisodio','centro']]
