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
CONFIGPATH=ROOTPATH+'configurations/'
USEDCONFIGPATH=CONFIGPATH+'used/'
OUTPATH=ROOTPATH+'predictions/'
METRICSPATH=ROOTPATH+'metrics/'
MODELSPATH=ROOTPATH+'models/'

"""FILES"""
ALLHOSPITFILE='ingresos2016_2018.csv'
ACGFILES={2016:'ing2016-2017Activos.csv',
          2017:'ing2017-2018Activos.csv',
          2018:'ing2018-2019Activos.csv'}
""" not "active" diagnoses:
ACGFILES={2016:'indispensable/2016.txt',
          2017:'indispensable/2017.txt',
          2018:'indispensable/2018.txt'}
"""
ACGINDPRIVFILES={2016:'ing2016-2017ActivosIndPriv.csv',
                  2017:'ing2017-2018ActivosIndPriv.csv',
                  2018:'ing2018-2019ActivosIndPriv.csv'}
ALLEDCFILES={2016:'additional_edcs2016.csv',
             2017:'additional_edcs2017.csv'}

FULLACGFILES={2016:'fullacgs2016.csv',
             2017:'fullacgs2017.csv'}

"""VERBOSITY SETTINGS"""
VERBOSE=True 
TRACEBACK=False #EXTREME VERBOSITY 



