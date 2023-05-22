#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:23:05 2023

@author: aolza
"""
import sys
import os
import configurations.utility as util
from python_settings import settings as config
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()

logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import joblib
from dataManipulation.dataPreparation import getData
import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
#%%
X,y=getData(2016)
X17,y18=getData(2017)
#%%
def read_older_ages(year,X):
    older_ages=pd.read_csv(config.INDISPENSABLEDATAPATH+f'base{year}_85omas.csv',sep=';')[['id','edad']]
    young=X.loc[~X.PATIENT_ID.isin(older_ages.id)]
    df=young.PATIENT_ID.to_frame()
    df['edad']=0
    df.rename(columns={'PATIENT_ID':'id'},inplace=True)
    older_ages=pd.concat([older_ages,df])
    return(older_ages)
older_ages=read_older_ages(2016, X)
older_ages2017=read_older_ages(2017, X17)
X['AGE_8589']=np.where(X.PATIENT_ID.isin(older_ages.id),older_ages.edad,0)
X['AGE_8589']=np.where(X['AGE_8589']<=89,1,0)

X17['AGE_8589']=np.where(X17.PATIENT_ID.isin(older_ages2017.id),older_ages2017.edad,0)
X17['AGE_8589']=np.where(X17['AGE_8589']<=89,1,0)
#%%
from pathlib import Path
if config.ALGORITHM=='linear':
    linear_filename='/home/aolza/Desktop/estratificacion/models/costCCS_parsimonious/linear_withcategory_8589.joblib'
    if Path(linear_filename).is_file():
        fit=joblib.load(linear_filename)
    else:
        linear=LinearRegression(n_jobs=-1)
        fit=linear.fit(X.filter(regex='AGE|FEMALE|CCS'), y[config.COLUMNS])
        modelname, modelfilename=util.savemodel(config, fit,name='linear_withcategory_8589', return_=True)
    
    mujeres=X17.loc[X17.FEMALE==1]
    hombres=X17.loc[X17.FEMALE==0]
    preds_muj=fit.predict(mujeres.filter(regex='AGE|FEMALE|CCS')).ravel()
    preds_hom=fit.predict(hombres.filter(regex='AGE|FEMALE|CCS')).ravel()
    
    preds_muj=pd.DataFrame(np.transpose(np.array([y18.loc[y18.PATIENT_ID.isin(mujeres.PATIENT_ID)].PATIENT_ID.values.ravel(),
                            preds_muj,
                            y18.loc[y18.PATIENT_ID.isin(mujeres.PATIENT_ID)].COSTE_TOTAL_ANO2.values.ravel(),
                           np.array([1]*len(preds_muj))])),
                           columns=['PATIENT_ID','PRED','OBS','FEMALE'])
    preds_hom=pd.DataFrame(np.transpose(np.array([y18.loc[y18.PATIENT_ID.isin(hombres.PATIENT_ID)].PATIENT_ID,
                            preds_hom,
                            y18.loc[y18.PATIENT_ID.isin(hombres.PATIENT_ID)].COSTE_TOTAL_ANO2,
                           np.array([0]*len(preds_hom))])),
                           columns=['PATIENT_ID','PRED','OBS','FEMALE'])
    #%%
        
else:
    logistic_filename='/home/aolza/Desktop/estratificacion/models/urgcmsCCS_parsimonious/logistic_withcategory_8589.joblib'
    if not Path(logistic_filename).is_file():
        logistic=LogisticRegression(n_jobs=-1)
        fit=logistic.fit(X.filter(regex='AGE|FEMALE|CCS'), np.where(y[config.COLUMNS]>=1,1,0))
        modelname, modelfilename=util.savemodel(config, fit,name='logistic_withcategory_8589', return_=True)
    else:
        fit=joblib.load(logistic_filename)
#%%
mujeres=X17.loc[X17.FEMALE==1]
hombres=X17.loc[X17.FEMALE==0]
if config.ALGORITHM=='linear':
    preds_muj=fit.predict(mujeres.filter(regex='AGE|FEMALE|CCS')).ravel()
    preds_hom=fit.predict(hombres.filter(regex='AGE|FEMALE|CCS')).ravel()
else:
    preds_muj=fit.predict_proba(mujeres.filter(regex='AGE|FEMALE|CCS'))[:,1].ravel()
    preds_hom=fit.predict_proba(hombres.filter(regex='AGE|FEMALE|CCS'))[:,1].ravel()

preds_muj=pd.DataFrame(np.transpose(np.array([y18.loc[y18.PATIENT_ID.isin(mujeres.PATIENT_ID)].PATIENT_ID.values.ravel(),
                        preds_muj,
                        y18.loc[y18.PATIENT_ID.isin(mujeres.PATIENT_ID)][config.COLUMNS].values.ravel(),
                       np.array([1]*len(preds_muj))])),
                       columns=['PATIENT_ID','PRED','OBS','FEMALE'])
preds_hom=pd.DataFrame(np.transpose(np.array([y18.loc[y18.PATIENT_ID.isin(hombres.PATIENT_ID)].PATIENT_ID,
                        preds_hom,
                        y18.loc[y18.PATIENT_ID.isin(hombres.PATIENT_ID)][config.COLUMNS].values.ravel(),
                       np.array([0]*len(preds_hom))])),
                       columns=['PATIENT_ID','PRED','OBS','FEMALE'])

#%%
def patient_selection(x,modeltype):
    x['top20k']=np.where(x.PATIENT_ID.isin(x.nlargest(20000,'PRED').PATIENT_ID),1,0)
    top10k_women=x.loc[x.FEMALE==1].nlargest(10000,'PRED')
    top10k_men=x.loc[x.FEMALE==0].nlargest(10000,'PRED')
    top1perc_women=x.loc[x.FEMALE==1].nlargest(int(0.01*len(x.loc[x.FEMALE==1])),'PRED')
    top1perc_men=x.loc[x.FEMALE==0].nlargest(int(0.01*len(x.loc[x.FEMALE==0])),'PRED')
    x['top10k_gender']=np.where(x.PATIENT_ID.isin(pd.concat([top10k_women,top10k_men]).PATIENT_ID),1,0)
    x['top1perc_gender']=np.where(x.PATIENT_ID.isin(pd.concat([top1perc_women,top1perc_men]).PATIENT_ID),1,0)
    
    if modeltype=='linear':
        top10k_women=x.loc[x.FEMALE==1].nlargest(10000,'OBS')
        top10k_men=x.loc[x.FEMALE==0].nlargest(10000,'OBS')
        top1perc_women=x.loc[x.FEMALE==1].nlargest(int(0.01*len(x.loc[x.FEMALE==1])),'OBS')
        top1perc_men=x.loc[x.FEMALE==0].nlargest(int(0.01*len(x.loc[x.FEMALE==0])),'OBS')
        
        x['should_be_selected_top20k']=np.where(x.PATIENT_ID.isin(x.nlargest(20000,'OBS').PATIENT_ID),1,0)
        x['should_be_selected_top10k_gender']=np.where(x.PATIENT_ID.isin(pd.concat([top10k_women,top10k_men]).PATIENT_ID),1,0)
        x['should_be_selected_top1perc_gender']=np.where(x.PATIENT_ID.isin(pd.concat([top1perc_women,top1perc_men]).PATIENT_ID),1,0)
        x['should_be_selected']=x['should_be_selected_top20k']
        
    else:
        x['should_be_selected']=np.where(x.OBS>=1,1,0)
        for col in ['top20k','top10k_gender','top1perc_gender']:
            x['should_be_selected_{col}']=x['should_be_selected_{col}']
    return x
from sklearn.metrics import confusion_matrix
def cm(x,col): return {key: val for key, val in zip(['tn', 'fp', 'fn', 'tp'],confusion_matrix(x[f'should_be_selected_{col}'],x[col]).ravel())} 
def threshold_muj_hom(x,col):
    c=cm(x.loc[x.FEMALE==1],col)
    print('Women c:' , c)
    vpp_women, vpn_women, sens_women, esp_women=c['tp']/(c['tp']+c['fp']),    c['tn']/(c['tn']+c['fn']),    c['tp']/(c['tp']+c['fn']),    c['tn']/(c['tn']+c['fp'])
    print('vpp_women,', vpp_women)
    c=cm(x.loc[x.FEMALE==0],col)
    vpp_men, vpn_men, sens_men, esp_men=c['tp']/(c['tp']+c['fp']),    c['tn']/(c['tn']+c['fn']),    c['tp']/(c['tp']+c['fn']),    c['tn']/(c['tn']+c['fp'])
    p=x.loc[x[col]==1]
    cp_women=p.loc[p.FEMALE==1].PRED.min()
    cp_men=p.loc[p.FEMALE==0].PRED.min()
    return(vpp_women, vpp_men, vpn_women, vpn_men, sens_women, sens_men, esp_women, esp_men, cp_women, cp_men)

def table(preds,modeltype):
    index=['global']
    if modeltype=='logistic':
        # Overall metrics: auc and brier
        from sklearn.metrics import roc_auc_score,brier_score_loss 
        def AUC_muj(x): return roc_auc_score(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def AUC_hom(x): return roc_auc_score(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        def brier_muj(x): return brier_score_loss(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def brier_hom(x): return brier_score_loss(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        
        overall_metrics=[[AUC_muj(preds),AUC_hom(preds),brier_muj(preds),brier_hom(preds)]]
        df_overall=pd.DataFrame(overall_metrics,columns=['AUC_women','AUC_men','brier_women','brier_men'],index=index)
    else:
        #Overall metrics: R2 and RMSE
        from sklearn.metrics import r2_score,mean_squared_error 
        def R2_muj(x): return r2_score(x.loc[x.FEMALE==1].OBS,x.loc[x.FEMALE==1].PRED)
        def R2_hom(x): return r2_score(x.loc[x.FEMALE==0].OBS,x.loc[x.FEMALE==0].PRED)
        def RSS_muj(x): return ((x.loc[x.FEMALE==1].OBS-x.loc[x.FEMALE==1].PRED)**2).sum()
        def RSS_hom(x): return ((x.loc[x.FEMALE==0].OBS-x.loc[x.FEMALE==0].PRED)**2).sum()
        def RMSE_muj(x): return mean_squared_error(x.loc[x.FEMALE==1].OBS,x.loc[x.FEMALE==1].PRED,squared=False)
        def RMSE_hom(x): return mean_squared_error(x.loc[x.FEMALE==0].OBS,x.loc[x.FEMALE==0].PRED,squared=False)
        
        overall_metrics=[[R2_muj(preds),R2_hom(preds),RMSE_muj(preds),RMSE_hom(preds),RSS_muj(preds),RSS_hom(preds)]]
        df_overall=pd.DataFrame(overall_metrics,columns=['R2_women','R2_men','RMSE_women','RMSE_men','RSS_women','RSS_men'],index=index)

    # PATIENT SELECTION:
    # We use three criteria: 
        # 1) Select the top 20k patients as positive, regardless of gender -> top20k
        # 2) Select the top 10k women and top 10k men -> top10k_gender
        # 3) Divide each prediction by the healthy equivalent per gender. Select the top 20k -> relative
   
    preds=patient_selection(preds,modeltype)
    
    # In either case, we use the same threshold-specific metrics
    threshold_metrics=[np.array([threshold_muj_hom(preds, 'top20k'),
                                 threshold_muj_hom(preds, 'top10k_gender'),
                                 threshold_muj_hom(preds, 'top1perc_gender')]).ravel()]
    
    threshold_metrics=[list(e)+list([100*preds.loc[preds.top20k==1].FEMALE.sum()/preds.top20k.sum()])
                       +list([100*preds.loc[preds.top1perc_gender==1].FEMALE.sum()/preds.top1perc_gender.sum()]) for e in threshold_metrics]
    
    df_threshold=pd.DataFrame(threshold_metrics,columns=['PPV_20k_women','PPV_20k_men','NPV_20k_women','NPV_20k_men',
                                                         'SENS_20k_women','SENS_20k_men','SPEC_20k_women','SPEC_20k_men',
                                                         'Cutpoint_top20k_women','Cutpoint_top20k_men',
                                                'PPV_10k_women','PPV_10k_men','NPV_10k_women','NPV_10k_men',
                                                'SENS_10k_women','SENS_10k_men','SPEC_10k_women','SPEC_10k_men',
                                                'Cutpoint_10k_women','Cutpoint_10k_men',
                                                'PPV_1perc_women','PPV_1perc_men','NPV_1perc_women','NPV_1perc_men',
                                                'SENS_1perc_women','SENS_1perc_men','SPEC_1perc_women','SPEC_1perc_men',
                                                'Cutpoint_1perc_women','Cutpoint_1perc_men',
                                                'Perc_top20k_women', 'Perc_top1perc_women'],
                 index=index)
    
    df=pd.concat([df_overall,df_threshold],axis=1)
    
    nmen=len(preds)-preds.FEMALE.sum()
    print('N women ',preds.FEMALE.sum())
    print('N men ',nmen)
    print('Prevalence women: ',np.where(preds.loc[preds.FEMALE==1].OBS,1,0).sum()/preds.FEMALE.sum())
    print('Prevalence men: ',np.where(preds.loc[preds.FEMALE==0].OBS,1,0).sum()/nmen)
    
    return df
#%%
allpreds=pd.concat([preds_muj,preds_hom])
allpreds.loc[allpreds.PRED<0,'PRED']=0
# r2_score(allpreds.OBS,allpreds.PRED)
table_linear=table(allpreds,config.ALGORITHM)
#%%
table_lin_round=table_linear.round(3)
cols=pd.Series([re.sub('_men|_women','',c) for c in table_linear]).drop_duplicates()
table=pd.DataFrame(index=table_lin_round.index)
for col in cols:
    table[col]=[c for c in table_lin_round.filter(regex=col).values]
print(table.T.to_latex())
#%%
preds_muj=patient_selection(preds_muj,config.ALGORITHM)
preds_hom=patient_selection(preds_hom,config.ALGORITHM)
def cm2(x,col):
    obs=np.where(x.PATIENT_ID.isin(x.nlargest(int(0.01*len(x)),'OBS').PATIENT_ID),1,0)
    return {key: val for key, val in zip(['tn', 'fp', 'fn', 'tp'],
                                                     confusion_matrix(obs,x[col]).ravel())} 
c=cm2(preds_muj,'top1perc_gender')
vpp_women, vpn_women, sens_women, esp_women=c['tp']/(c['tp']+c['fp']),    c['tn']/(c['tn']+c['fn']),    c['tp']/(c['tp']+c['fn']),    c['tn']/(c['tn']+c['fp'])

c=cm2(preds_hom,'top1perc_gender')

#%%
