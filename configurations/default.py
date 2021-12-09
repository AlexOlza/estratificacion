#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:14:16 2021

@author: aolza
"""
SEED=42 #random seed
"""LOCATIONS"""
PROJECT='estratificacion'
ROOTPATH='/home/aolza/Desktop/estratificacion/'
DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
INDISPENSABLEDATAPATH=DATAPATH+'indispensable/'
"""these will be overwritten for cluster """
CONFIGPATH=ROOTPATH+'configurations/local/'
USEDCONFIGPATH=CONFIGPATH+'used/'
OUTPATH=ROOTPATH+'predictions/'
MODELSPATH=ROOTPATH+'models/'

"""FILES"""
ALLHOSPITFILE='ingresos2016_2018.csv'
ACGFILES={2016:'ing2016-2017Activos.csv',
          2017:'ing2017-2018Activos.csv',
          2018:'ing2018-2019Activos.csv'}
ACGINDPRIVFILES={2016:'ing2016-2017ActivosIndPriv.csv',
                  2017:'ing2017-2018ActivosIndPriv.csv',
                  2018:'ing2018-2019ActivosIndPriv.csv'}
"""PREDICTORS: They will be different for each script."""
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT|INDICE_PRIVACION'
PREDICTORS=True
INDICEPRIVACION=True
COLUMNS=[]
PREVIOUSHOSP=[]
EXCLUDE=[]

"""VERBOSITY SETTINGS"""
VERBOSE=True 
TRACEBACK=False #EXTREME VERBOSITY 
