#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:12:55 2022

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import numpy as np
from modelEvaluation.predict import predict


# X,y=getData(2017)
# pastX,pasty=getData(2016)
# #%%
# Xacg,yacg=getData(2017,
#                   CCS=False,PHARMACY=False,BINARIZE_CCS=False,
#                   predictors=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_(?!NUR11|RES10)|HOSDOM|FRAILTY|RXMG_(?!ZZZX000)|INGRED_14GT',
#                   )
# pastXacg,pastyacg=getData(2016, 
#                   CCS=False,PHARMACY=False,BINARIZE_CCS=False,
#                   predictors=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_(?!NUR11|RES10)|HOSDOM|FRAILTY|RXMG_(?!ZZZX000)|INGRED_14GT',
#                   )
#%%
# assert False
ccsmodel='linear20230201_122433' if config.ALGORITHM=='linear' else 'logistic20230201_121805'
acgmodel='linear20221205_132708' if config.ALGORITHM=='linear' else 'logistic20220705_155354'
exp='cost_ACG' if config.ALGORITHM=='linear' else 'urgcms_excl_nbinj'
predsCCS=predict(ccsmodel, config.EXPERIMENT, 2018,
                 # X=X,y=y,pastX=pastX,pasty=pasty
                 )
predsACG=predict(acgmodel, exp,2018,
                 # X=Xacg,y=yacg,pastX=pastXacg,pasty=pastyacg
                 )

#%%
predsACG=predsACG[0]; predsCCS=predsCCS[0]

if config.ALGORITHM=='linear':
    predsACG.loc[predsACG.PRED<0,'PRED']=0
    predsCCS.loc[predsCCS.PRED<0,'PRED']=0
#%%
acg20k=predsACG.nlargest(20000,'PRED')
ccs20k=predsCCS.nlargest(20000,'PRED')
#%%
len(set(acg20k.PATIENT_ID).intersection(set(ccs20k.PATIENT_ID)))/20000
#coste 13220 (66%)
#ing 12017 (60%)

#%%
import joblib as job
acg_model=job.load('/home/alex/Desktop/estratificacion/models/cost_ACG/linear20221205_132708.joblib')
#%%
predsACG.PRED.corr(predsCCS.PRED, method='spearman') #ingreso 0.9016; coste 0.921 not 0.9107

if config.ALGORITHM=='linear':
    s=(predsACG.PRED/predsACG.PRED.mean()).to_frame()
    s['perc']=s.rank(pct=True)
    sCCS=(predsCCS.PRED/predsCCS.PRED.mean()).to_frame()
    sCCS['perc']=sCCS.rank(pct=True)
    print('Mean cost: ',predsACG.PRED.mean(),predsCCS.PRED.mean())
    print('Perc 99: ',predsACG.PRED.quantile(0.99),predsCCS.PRED.quantile(0.99))
    print('Perc 95: ',predsACG.PRED.quantile(0.95),predsCCS.PRED.quantile(0.95))
    print('IP 6.2 Perc: ',s.loc[s.PRED.round(3)==6.200].perc.median(),sCCS.loc[sCCS.PRED.round(3)==6.200].perc.median())
    # print('Mean cost: ',predsACG.PRED.mean())