#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()

"""THIS EMULATES 'from experiment import *' USING IMPORTLIB 
info: 
    https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
"""
mdl=importlib.import_module(experiment,package='estratificacion')
# is there an __all__?  if so respect it
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    # otherwise we import all names that don't begin with _
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
globals().update({k: getattr(mdl, k) for k in names}) #brings everything into namespace

from configurations.default import *
import os
import sys

MODELPATH=MODELSPATH+EXPERIMENT+'/'
ALGORITHM='HGB'
CONFIGNAME='configHGB.py'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT)
TRACEBACK=False

""" SETTINGS FOR THE RANDOM FOREST """
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
IMPORTS=[]
# def agrupar(datos):
#     edad=[p for p in np.array(datos.filter(regex=r'AGE').columns)]
#     edad.append('AGE_85GT')
#     datos['AGE_85GT']=0
#     datos.iloc[datos[edad].max(1)==0, datos.columns.get_loc('AGE_85GT')]=1
#     datos['edad']=datos[edad].idxmax(1)
#     datos['edad']=datos.edad.astype("category").cat.codes
    
#     ACG=[p for p in np.array(datos.filter(regex=r'ACG_').columns)]
#     ACG.append('NINGUNO')
#     datos['NINGUNO']=0
#     datos.iloc[datos[ACG].max(1)==0, datos.columns.get_loc('NINGUNO')]=1
#     datos['ACG']=datos[ACG].idxmax(1)
#     datos.ACG=datos.ACG.astype("category").cat.codes
#     return(datos)
# binaryPredictors=[p for p in 
#             np.array(datos.filter(regex=r'FEMALE|ACGRUB|EDC_|HOSDOM|FRAILTY|RXMG').columns)]


# predictors=binaryPredictors+['edad','ACG']
# """AGRUPO LAS VARIABLES CATEGORICAS- EDAD Y ACG"""

# datos=agrupar(datos)
FOREST=HistGradientBoostingClassifier(loss='auto', max_bins=255,
                                   # categorical_features=cat,
                                   monotonic_cst=None,
                                   warm_start=False, early_stopping=False,
                                   scoring='loss', validation_fraction=0.1,
                                   n_iter_no_change=10, tol=1e-07,
                                   random_state=SEED)


learning_rate=[0.01,0.1,0.3,0.5,0.5,1.0]
max_iter=[500,1000,1500,2500,5000] #number of trees
max_leaf_nodes=[15,30,45,60,None]
max_depth=[None, 10,20,30,50]
min_samples_leaf=[10,20,40,80]
l2_regularization=[0.0,0.1,0.5,1.0]

RANDOM_GRID = {'learning_rate':learning_rate,
               'max_iter':max_iter,
               'max_leaf_nodes':max_leaf_nodes,
               'max_depth':max_depth,
               'min_samples_leaf':min_samples_leaf,
               'l2_regularization':l2_regularization}


N_ITER=50
CV=3
