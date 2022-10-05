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
from modelEvaluation.regression.sex_functions import *
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

year=2018#int(input('YEAR TO PREDICT: '))
filename=config.PREDPATH+'/sexSpecificOddsContributions.csv'
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
    
    modeloH=job.load(config.MODELPATH+'linearHombres.joblib')
    
    stderrH=beta_std_error(modeloH, Xhom, yhom[config.COLUMNS])

    
   
else:
    oddsContrib=pd.read_csv(filename)
    
    