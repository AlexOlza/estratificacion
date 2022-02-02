#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:14:16 2021

@author: aolza
"""
#from configurations.almeria import *
SEED=42 #random seed
"""LOCATIONS"""
PROJECT='estratificacion'
ROOTPATH='/home/aolza/Desktop/estratificacion/'
DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
INDISPENSABLEDATAPATH=DATAPATH+'indispensable/'
CONFIGPATH=ROOTPATH+'configurations/'
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
#Exclude patients from Tolosaldea and Errioxa because they receive
#most of their care outside of Osakidetza.
EXCLUDEOSI=['OS16','OS22'] 

"""VERBOSITY SETTINGS"""
VERBOSE=True 
TRACEBACK=False #EXTREME VERBOSITY 



