#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:09:11 2023

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
import re
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
figurepath='/home/aolza/Desktop/estratificacion/figures/gender_article_draft'
CCS=eval(input('CCS? True/False: '))
ccs='CCS' if CCS else 'ACG'

if CCS:
    logistic_modelpath=config.ROOTPATH+'models/urgcmsCCS_parsimonious/'
    linear_modelpath=config.ROOTPATH+'models/costCCS_parsimonious/'
    
    logistic_modelname='logistic20230324_111354'
    linear_modelname='linear20230324_130625'
    
    logistic_predpath=re.sub('models','predictions',logistic_modelpath)
    linear_predpath=re.sub('models','predictions',linear_modelpath)
else: #ACG
    logistic_modelpath=config.ROOTPATH+'models/urgcms_excl_nbinj/'
    linear_modelpath=config.ROOTPATH+'models/cost_ACG/'
    
    logistic_modelname='logistic20220705_155354'
    linear_modelname='linear20221018_103900'
    
    logistic_predpath=re.sub('models','predictions',logistic_modelpath)
    linear_predpath=re.sub('models','predictions',linear_modelpath)
logistic_global_model=joblib.load(logistic_modelpath+f'{logistic_modelname}.joblib')
linear_global_model=joblib.load(linear_modelpath+f'{linear_modelname}.joblib')
logistic_women_model=joblib.load(logistic_modelpath+f'logisticMujeres.joblib')
linear_women_model=joblib.load(linear_modelpath+f'linearMujeres.joblib')
logistic_men_model=joblib.load(logistic_modelpath+f'logisticHombres.joblib')
linear_men_model=joblib.load(linear_modelpath+f'linearHombres.joblib')
logistic_sameprev_model=joblib.load(logistic_modelpath+'logistic_gender_balanced.joblib')


log_global_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_global_model.feature_names_in_, logistic_global_model.coef_[0])},
                                        orient='index')
lin_global_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(linear_global_model.feature_names_in_, linear_global_model.coef_[0])},
                                        orient='index')
log_sameprev_coefs=pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_sameprev_model.feature_names_in_, logistic_sameprev_model.coef_[0])},
                                        orient='index')
print('logistic global female coef:',log_global_coefs.loc['FEMALE'].values )
print('logistic same prevalence female coef:',log_sameprev_coefs.loc['FEMALE'].values )
print('linear global female coef:', lin_global_coefs.loc['FEMALE'].values  )
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
        x['should_be_selected_']=x['should_be_selected_top20k']
        
    else:
        x['should_be_selected_']=np.where(x.OBS>=1,1,0)
        Nmuj_equitativo=int(0.16*len(x.loc[(x.FEMALE==1) & (x['should_be_selected_']==1)]))
        Nhom_equitativo=int(0.16*len(x.loc[(x.FEMALE==0) & (x['should_be_selected_']==1)]))
        muj_equit=x.loc[x.FEMALE==1].nlargest(Nmuj_equitativo,'PRED')
        hom_equit=x.loc[x.FEMALE==0].nlargest(Nhom_equitativo,'PRED')
        x['proporcion_equitativa']=np.where(x.PATIENT_ID.isin(pd.concat([muj_equit,hom_equit]).PATIENT_ID),1,0)
        for col in ['top20k','top10k_gender','top1perc_gender','proporcion_equitativa']:
            x[f'should_be_selected_{col}']=x['should_be_selected_'].copy()
    return x
from sklearn.metrics import confusion_matrix
def cm(x,col): return {key: val for key, val in zip(['tn', 'fp', 'fn', 'tp'],confusion_matrix(x[f'should_be_selected_'],x[col]).ravel())} 
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
    
    if modeltype=='linear':
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
    else:
        threshold_metrics=[np.array([threshold_muj_hom(preds, 'top20k'),
                                     threshold_muj_hom(preds, 'top10k_gender'),
                                     threshold_muj_hom(preds, 'top1perc_gender'),
                                     threshold_muj_hom(preds, 'proporcion_equitativa')]).ravel()]
        
        threshold_metrics=[list(e)+list([100*preds.loc[preds.top20k==1].FEMALE.sum()/preds.top20k.sum()])
                           +list([100*preds.loc[preds.top1perc_gender==1].FEMALE.sum()/preds.top1perc_gender.sum()])
                           +list([100*preds.loc[preds.proporcion_equitativa==1].FEMALE.sum()/preds.proporcion_equitativa.sum()])
                           for e in threshold_metrics]
        
        df_threshold=pd.DataFrame(threshold_metrics,columns=['PPV_20k_women','PPV_20k_men','NPV_20k_women','NPV_20k_men',
                                                             'SENS_20k_women','SENS_20k_men','SPEC_20k_women','SPEC_20k_men',
                                                             'Cutpoint_top20k_women','Cutpoint_top20k_men',
                                                    'PPV_10k_women','PPV_10k_men','NPV_10k_women','NPV_10k_men',
                                                    'SENS_10k_women','SENS_10k_men','SPEC_10k_women','SPEC_10k_men',
                                                    'Cutpoint_10k_women','Cutpoint_10k_men',
                                                    'PPV_1perc_women','PPV_1perc_men','NPV_1perc_women','NPV_1perc_men',
                                                    'SENS_1perc_women','SENS_1perc_men','SPEC_1perc_women','SPEC_1perc_men',
                                                    'Cutpoint_1perc_women','Cutpoint_1perc_men',
                                                    'PPV_equit_women','PPV_equit_men','NPV_equit_women','NPV_equit_men',
                                                    'SENS_equit_women','SENS_equit_men','SPEC_equit_women','SPEC_equit_men',
                                                    'Cutpoint_equit_women','Cutpoint_equit_men',
                                                    'Perc_top20k_women', 'Perc_top1perc_women','Perc_equit_women'],
                     index=index)
    
    df=pd.concat([df_overall,df_threshold],axis=1)
    
    nmen=len(preds)-preds.FEMALE.sum()
    print('N women ',preds.FEMALE.sum())
    print('N men ',nmen)
    print('Prevalence women: ',np.where(preds.loc[preds.FEMALE==1].OBS,1,0).sum()/preds.FEMALE.sum())
    print('Prevalence men: ',np.where(preds.loc[preds.FEMALE==0].OBS,1,0).sum()/nmen)
    
    return df
def concat_preds(file1,file2):
    muj=pd.read_csv(file1)
    muj['FEMALE']=1
    hom=pd.read_csv(file2)
    hom['FEMALE']=0
    return pd.concat([muj,hom])
#%%
allpreds=concat_preds(logistic_predpath+f'{logistic_modelname}_Mujeres_calibrated_2018.csv',
                      logistic_predpath+f'{logistic_modelname}_Hombres_calibrated_2018.csv') 
allpreds.loc[allpreds.PRED<0,'PRED']=0
# r2_score(allpreds.OBS,allpreds.PRED)
table_logistic=table(allpreds,'logistic')
#%%
allpreds_lin=concat_preds(linear_predpath+f'{linear_modelname}_Mujeres__2018.csv',
                      linear_predpath+f'{linear_modelname}_Hombres__2018.csv') 
allpreds_lin.loc[allpreds_lin.PRED<0,'PRED']=0
# r2_score(allpreds.OBS,allpreds.PRED)
table_linear=table(allpreds_lin,'linear')
#%%
table_log_round=table_logistic.round(3)
cols=pd.Series([re.sub('_men|_women','',c) for c in table_logistic]).drop_duplicates()
tablelog=pd.DataFrame(index=table_log_round.index)
for col in cols:
    tablelog[col]=[c for c in table_log_round.filter(regex=col).values]
print(tablelog.T.to_latex())
#%%
table_lin_round=table_linear.round(3)
cols=pd.Series([re.sub('_men|_women','',c) for c in table_linear]).drop_duplicates()
tablelin=pd.DataFrame(index=table_lin_round.index)
for col in cols:
    tablelin[col]=[c for c in table_lin_round.filter(regex=col).values]
print(tablelin.T.to_latex())
#%%
""" Proporcion equitativa coste: 
    Porcentaje de  mujeres entre los 20k de mayor coste observado
"""
table_linear['Prop_equit']=100*allpreds_lin.loc[allpreds_lin['should_be_selected_top20k']==1].FEMALE.sum()/20000
