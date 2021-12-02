#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CREA NUEVAS BASES DE DATOS CON NÚMERO DE INGRESOS
Created on Mon Jun 28 12:36:49 2021

@author: aolza
"""
import config
import pandas as pd
import datetime
ing1617=pd.read_csv(config.INDISPENSABLEDATAPATH+'ingresos_2016_2017.csv')
ing1819=pd.read_csv(config.INDISPENSABLEDATAPATH+'ingresos_2018_2019.csv')

# ing['Prioridad'].unique() -> URGENTE, PROGRAMADO

df=pd.DataFrame(columns=['PATIENT_ID','ingresoPrevioUrg','ingresoPrevioProg','ingresoUrg','ingresoProg'])

ing=pd.concat([ing1617,ing1819])
ing['Fecha']=pd.to_datetime(ing['Fecha (ingreso)']).dt.year
fechas= ing['Fecha'].unique()
print(ing['Fecha'].unique())
d1819,d1718,d1617=dict(),dict(),dict()
i=0
for paciente in ing['﻿Id Paciente'].unique():
    pac=ing[ing['﻿Id Paciente']==paciente]
    ingreso16=pac[pac['Fecha']==2016]
    ingreso16Urgente=ingreso16[ingreso16['Prioridad']=='URGENTE']
    ingreso16Programado=ingreso16[ingreso16['Prioridad']=='PROGRAMADO']
    ingreso17=pac[pac['Fecha']==2017]
    ingreso17Urgente=ingreso17[ingreso17['Prioridad']=='URGENTE']
    ingreso17Programado=ingreso17[ingreso17['Prioridad']=='PROGRAMADO']
    ingreso18=pac[pac['Fecha']==2018]
    ingreso18Urgente=ingreso18[ingreso18['Prioridad']=='URGENTE']
    ingreso18Programado=ingreso18[ingreso18['Prioridad']=='PROGRAMADO']
    ingreso19=pac[pac['Fecha']==2019]
    ingreso19Urgente=ingreso19[ingreso19['Prioridad']=='URGENTE']
    ingreso19Programado=ingreso19[ingreso19['Prioridad']=='PROGRAMADO']
    
    d1617[i]=[paciente,len(ingreso16Urgente),len(ingreso16Programado),
          len(ingreso17Urgente),len(ingreso17Programado)]
    d1718[i]=[paciente,len(ingreso17Urgente),len(ingreso17Programado),
          len(ingreso18Urgente),len(ingreso18Programado)]
    d1819[i]=[paciente,len(ingreso18Urgente),len(ingreso18Programado),
          len(ingreso19Urgente),len(ingreso19Programado)]
    print(d1819[i],d1718[i],d1617[i])
    i+=1

for d,yr in zip([d1617,d1718,d1819],['1617','1718','1819']):
    df=pd.DataFrame.from_dict(d, orient='index',
                              columns=['PATIENT_ID','ingresoPrevioUrg','ingresoPrevioProg','ingresoUrg','ingresoProg'])
    df.to_csv(config.DATAPATH+'ing{0}.csv'.format(yr))



