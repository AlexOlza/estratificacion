#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:16:18 2022

@author: alex
"""
cluster=False
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
import pandas as pd
from modelEvaluation.predict import predict
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from modelEvaluation.compare import performance
from dataManipulation.dataPreparation import getData
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import os
config.MODELPATH=os.path.join(config.MODELPATH,'nested_logistic/') if not 'nested_logistic' in config.MODELPATH else config.MODELPATH
config.PREDPATH=os.path.join(config.PREDPATH,'nested_logistic/') if not 'nested_logistic' in config.PREDPATH else config.PREDPATH
for direct in [config.MODELPATH, config.PREDPATH]:
    if not os.path.exists(direct):
        os.makedirs(direct)
        print('new dir ', direct)
def fit_models(variables:dict,X, y, 
               estimator=LogisticRegression(penalty='none',max_iter=1000,verbose=0) ):
    for key, val in variables.items():
        print('STARTING ',key, val)
        if not val:
            continue
        if Path(os.path.join(config.MODELPATH,f'{key}.joblib')).is_file(): #the model is already there
            continue
        Xx=X.filter(regex=val, axis=1)
        
        if key=='GMA_peso_ip_categorizado':
            Xx=categorize_ip(Xx)
        
        print('Number of predictors: ',len(Xx.columns))
        t0=time()
        fit=estimator.fit(Xx, y)
        print('fitting time: ',time()-t0)

        util.savemodel(config, fit, name='{0}'.format(key))
def categorize_ip(X):
    dff=X.copy()
    quantiles=dff['GMA_peso-ip'].quantile([0.5,0.8,0.95,0.99])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.50],2,1)
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.80],3,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.95],4,dff['complejidad'])
    dff['complejidad']=np.where(dff['GMA_peso-ip']>=quantiles[0.99],5,dff['complejidad'])
    dff=pd.concat([dff, pd.get_dummies(dff.complejidad)],axis=1)
    dff.rename(columns={i:f'GMA_peso_quantile_{i}' for i in [1,2,3,4,5]},inplace=True)
    dff.drop(['complejidad'], axis=1,inplace=True)
    return dff
#%%
np.random.seed(config.SEED)

indices=["GMA_num_patol",
       "GMA_num_sist","GMA_peso-ip",
       "GMA_riesgo-muerte"]
patologias=["GMA_dm","GMA_ic","GMA_epoc",
       "GMA_hta","GMA_depre","GMA_vih","GMA_c_isq","GMA_acv",
       "GMA_irc","GMA_cirros",
       "GMA_osteopor","GMA_artrosis",
       "GMA_artritis","GMA_demencia","GMA_dolor_cron"]
additional_columns=indices+patologias


basico='PATIENT_ID|FEMALE|AGE_[0-9]+$|CCS|PHARMA|'
variables={ 'CCSPHARMA':basico[:-1],
            'GMA_peso_ip':basico+'GMA_peso-ip',
            'GMA_peso_ip_categorizado':basico+'GMA_peso-ip',
            'GMA_31cat':basico+'GMA_[0-9]+$',
            'GMA_indices': basico+ '|'.join(indices),
            'GMA_patologias':basico+ '|'.join(patologias),
            'GMA_indices_patologias': basico+ '|'.join(indices+patologias),
            'GMA_31cat_indices_patologias':basico+'|'.join(['GMA_[0-9]+$']+indices+patologias)
            }
variables_7cat={'GMA_7cat':basico+'GMA_[0-9]+$',
           'GMA_7cat_indices_patologias':basico+'|'.join(['GMA_[0-9]+$']+indices+patologias)}
variables_4cat={'GMA_4cat':basico+'GMA_[0-9]+$',
           'GMA_4cat_indices_patologias':basico+'|'.join(['GMA_[0-9]+$']+indices+patologias)}
                   
drop=[0,1,2]
vardict=[variables,variables_7cat,variables_4cat]
table=pd.DataFrame()

for drop_,vardict_ in zip(drop,vardict):
    if cluster:
        X,y=getData(2016,
                    CCS=True,
                    PHARMACY=True,
                    BINARIZE_CCS=True,
                    GMA=True,
                    GMACATEGORIES=True,
                    GMA_DROP_DIGITS=drop_,
                    additional_columns=additional_columns)
        # assert False
        to_drop=['PATIENT_ID','ingresoUrg']
        for c in to_drop:
            try:
                X.drop(c,axis=1,inplace=True)
                util.vprint('dropping col ',c)
            except:
                pass
                util.vprint('pass')
        
        y=np.where(y[config.COLUMNS]>=1,1,0)
        y=y.ravel()
        
        
        print('Sample size ',len(X), 'positive: ',sum(y))
    #%%
    
        fit_models(vardict_, X, y)


#%%
    if not cluster:
        X, y=getData(2017,
                    CCS=True,
                    PHARMACY=True,
                    BINARIZE_CCS=True,
                    GMA=True,
                    GMACATEGORIES=True,
                    GMA_DROP_DIGITS=drop_,
                    additional_columns=additional_columns)
        
        PATIENT_ID=X.PATIENT_ID
        if hasattr(config, 'target_binarizer'):
            y=config.target_binarizer(y)
        else:
            y=pd.Series(np.where(y[config.COLUMNS]>0,1,0).ravel(),name=config.COLUMNS[0])
           
        y=pd.concat([y, PATIENT_ID], axis=1) if not 'PATIENT_ID' in y else y
        #%%
        
        
        for key, val in vardict_.items():
            Xx=X.copy()
            Xx=Xx.filter(regex=val, axis=1)
            if key=='GMA_peso_ip_categorizado':
                Xx=categorize_ip(Xx)

            if not val:
                continue
            try:
                probs,_=predict(key,experiment_name=config.EXPERIMENT,year=2018,
                                  X=Xx, y=y)
                auc=roc_auc_score(probs.OBS,probs.PRED)
                recall, ppv, spec, newpred = performance(obs=probs.OBS, pred=probs.PRED, K=20000)
                
                brier=brier_score_loss(y_true=probs.OBS, y_prob=probs.PRED)
                ap=average_precision_score(probs.OBS,probs.PRED)
                table=pd.concat([table,
                                 pd.DataFrame.from_dict({'Model':[key], 'AUC':[ auc], 'AP':[ap],
                                  'R@20k': [recall], 'PPV@20K':[ppv], 
                                  'Brier':[brier]})])
                probs['TOP20k']=newpred
            except:
                print(key , 'Failed')
    
    #%%
print(table.to_markdown(index=False))
#%%
if not cluster:
    from matplotlib import pyplot as plt
    import joblib
    for m in ['GMA_31cat_indices_patologias','GMA_7cat_indices_patologias','GMA_4cat_indices_patologias',
              'GMA_31cat','GMA_7cat','GMA_4cat']:
        model=joblib.load(config.MODELPATH+f'{m}.joblib')
        coefs=pd.DataFrame({name:[coef] for name, coef in zip(model.feature_names_in_,model.coef_[0])}).T
        coefs.rename(columns={0:'beta'}, inplace=True)
        gmacoefs=coefs.loc[coefs.index.str.startswith('GMA')].sort_index()
        ccscoefs=coefs.loc[coefs.index.str.startswith('CCS')].nlargest(len(gmacoefs),columns='beta')
        fig,ax=plt.subplots(figsize=(40,30))
        gmacoefs.plot(ax=ax, use_index=False, title=m)
        ccscoefs.plot(ax=ax)
        x=np.arange(len(gmacoefs))
        y=gmacoefs.beta.values
        z=ccscoefs.beta.values
        for i, txt in enumerate(gmacoefs.index):
            ax.annotate(txt, (x[i], y[i]),fontsize=36)
        for i, txt in enumerate(ccscoefs.index):
            ax.annotate(txt, (x[i], z[i]),fontsize=26,rotation=270)