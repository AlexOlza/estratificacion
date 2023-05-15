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
figurepath='/home/aolza/Desktop/estratificacion/figures/gender_article_draft/'
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
        
        prop_equit=100*x.loc[x['should_be_selected_top20k']==1].FEMALE.sum()/20000
        muj_forzar=x.loc[x.FEMALE==1].nlargest(int(10000*prop_equit/(100-prop_equit)),'PRED')
        hom_forzar=x.loc[x.FEMALE==0].nlargest(10000,'PRED')
        x['forzarproporcion']=np.where(x.PATIENT_ID.isin(pd.concat([muj_forzar,hom_forzar]).PATIENT_ID),1,0)
        muj_forzar=x.loc[x.FEMALE==1].nlargest(int(10000*prop_equit/(100-prop_equit)),'OBS')
        hom_forzar=x.loc[x.FEMALE==0].nlargest(10000,'OBS')
        
        x['should_be_selected_forzarproporcion']=np.where(x.PATIENT_ID.isin(pd.concat([muj_forzar,hom_forzar]).PATIENT_ID),1,0)
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
        muj_forzar=x.loc[x.FEMALE==1].nlargest(int(10000*Nmuj_equitativo/Nhom_equitativo),'PRED')
        hom_forzar=x.loc[x.FEMALE==0].nlargest(10000,'PRED')
        x['forzarproporcion']=np.where(x.PATIENT_ID.isin(pd.concat([muj_forzar,hom_forzar]).PATIENT_ID),1,0)
        
        for col in ['top20k','top10k_gender','top1perc_gender','proporcion_equitativa','forzarproporcion']:
            x[f'should_be_selected_{col}']=x['should_be_selected_'].copy()
    return x
from sklearn.metrics import confusion_matrix
def cm(x,col): return {key: val for key, val in zip(['tn', 'fp', 'fn', 'tp'],confusion_matrix(x[f'should_be_selected_{col}'],x[col]).ravel())} 
def threshold_muj_hom(x,col):
    c=cm(x.loc[x.FEMALE==1],col)
    print(f'Women c {col}:' , c)
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
                                     threshold_muj_hom(preds, 'top1perc_gender'),
                                     threshold_muj_hom(preds, 'forzarproporcion')]).ravel()]
        
        threshold_metrics=[list(e)+list([100*preds.loc[preds.top20k==1].FEMALE.sum()/preds.top20k.sum()])
                           +list([100*preds.loc[preds.top1perc_gender==1].FEMALE.sum()/preds.top1perc_gender.sum()])
                           +list([preds.loc[preds.forzarproporcion==1].FEMALE.sum()])
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
                                                    'PPV_forzarproporcion_women','PPV_forzarproporcion_men','NPV_forzarproporcion_women','NPV_forzarproporcion_men',
                                                    'SENS_forzarproporcion_women','SENS_forzarproporcion_men','SPEC_forzarproporcion_women','SPEC_forzarproporcion_men',
                                                    'Cutpoint_forzarproporcion_women','Cutpoint_forzarproporcion_men',
                                                    'Perc_top20k_women', 'Perc_top1perc_women', 'N_forzarproporcion_women'],
                     index=index)
    else:
        threshold_metrics=[np.array([threshold_muj_hom(preds, 'top20k'),
                                     threshold_muj_hom(preds, 'top10k_gender'),
                                     threshold_muj_hom(preds, 'top1perc_gender'),
                                     threshold_muj_hom(preds, 'proporcion_equitativa'),
                                     threshold_muj_hom(preds, 'forzarproporcion')]).ravel()]
        
        threshold_metrics=[list(e)+list([100*preds.loc[preds.top20k==1].FEMALE.sum()/preds.top20k.sum()])
                           +list([100*preds.loc[preds.top1perc_gender==1].FEMALE.sum()/preds.top1perc_gender.sum()])
                           +list([100*preds.loc[preds.proporcion_equitativa==1].FEMALE.sum()/preds.proporcion_equitativa.sum()])
                           +list([preds.loc[preds.forzarproporcion==1].FEMALE.sum()])
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
                                                    'PPV_forzarproporcion_women','PPV_forzarproporcion_men','NPV_forzarproporcion_women','NPV_forzarproporcion_men',
                                                    'SENS_forzarproporcion_women','SENS_forzarproporcion_men','SPEC_forzarproporcion_women','SPEC_forzarproporcion_men',
                                                    'Cutpoint_forzarproporcion_women','Cutpoint_forzarproporcion_men',
                                                    'Perc_top20k_women', 'Perc_top1perc_women','Perc_equit_women',
                                                    'N_forzarproporcion_women'],
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

allpreds_sep=concat_preds(logistic_predpath+'logisticMujeres_calibrated_2018.csv',
                      logistic_predpath+'logisticHombres_calibrated_2018.csv') 
allpreds_sep.loc[allpreds.PRED<0,'PRED']=0
allpreds_sep=patient_selection(allpreds_sep, 'logistic')
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
    Proporción equitativa ingreso:
    Porcentaje de mujeres entre las personas que realmente ingresan (Perc equit en la tabla)
"""
table_linear['Prop_equit']=100*allpreds_lin.loc[allpreds_lin['should_be_selected_top20k']==1].FEMALE.sum()/20000

allpreds.loc[allpreds.FEMALE==1].should_be_selected_.sum()/allpreds.should_be_selected_.sum()
#%%
"""Porcentaje de mujeres entre los top 20k dividido por estratos:
    Dentro de esos 20000 pacientes los hombres siguen ocupando posiciones
    más altas de coste observado / predicho / probabilidad de ingreso predicha
"""
dataframe=pd.DataFrame()
# Probabilidad de ingreso
dataframe['Prob. ingreso']=[allpreds.nlargest(5000,'PRED').FEMALE.sum()*100/5000,
allpreds.nlargest(10000,'PRED').nsmallest(5000,'PRED').FEMALE.sum()*100/5000,
allpreds.nlargest(20000,'PRED').nsmallest(10000,'PRED').FEMALE.sum()*100/10000,
allpreds.nlargest(20000,'PRED').FEMALE.sum()*100/20000]

# Coste predicho
dataframe['Coste predicho']=[allpreds_lin.nlargest(5000,'PRED').FEMALE.sum()*100/5000,
allpreds_lin.nlargest(10000,'PRED').nsmallest(5000,'PRED').FEMALE.sum()*100/5000,
allpreds_lin.nlargest(20000,'PRED').nsmallest(10000,'PRED').FEMALE.sum()*100/10000,
allpreds_lin.nlargest(20000,'PRED').FEMALE.sum()*100/20000]

# Coste observado
dataframe['Coste observado']=[allpreds_lin.nlargest(5000,'OBS').FEMALE.sum()*100/5000,
allpreds_lin.nlargest(10000,'OBS').nsmallest(5000,'OBS').FEMALE.sum()*100/5000,
allpreds_lin.nlargest(20000,'OBS').nsmallest(10000,'OBS').FEMALE.sum()*100/10000,
allpreds_lin.nlargest(20000,'OBS').FEMALE.sum()*100/20000
]

dataframe.index=['Primeros 5000', 'Siguientes 5000', 'Siguientes 10000', 'Primeros 20000']

print(dataframe)
#%%
X,y=getData(2017)
#%%
X['AGE_older_than_85']=np.where(X.filter(regex='AGE').sum(axis=1)==0,1,0)
X['AGE_younger_than_54']=X['AGE_0004']+X['AGE_0511']+X['AGE_1217']+X['AGE_1834']+X['AGE_3544']+X['AGE_4554']
X_age=X.drop(['AGE_0004','AGE_0511','AGE_1217','AGE_1834','AGE_3544','AGE_4554'],axis=1)
#%%
from matplotlib import ticker
# fig,(ax3,ax4)=plt.subplots(1,2,figsize=(20,6))
# fig2,(axppv,axppv2)=plt.subplots(1,2,figsize=(20,6))
colorss=[['blue','red', 'black'],['lightblue','orange', 'black']]
for intervention,colors in zip(['top20k','top10k_gender'],colorss):
    fig,(ax3,ax4)=plt.subplots(1,2,figsize=(20,6))
    fig2,(axppv,axppv2)=plt.subplots(1,2,figsize=(20,6))
    for allpreds_,modelo,ls in zip([allpreds,allpreds_sep],['global','separate'],['-','--']):
        df_percentages_full=pd.DataFrame()
        for col in X_age.filter(regex='AGE').columns:
            Xx=X_age.loc[X_age[col]==1]
            preds=allpreds_.loc[allpreds_.PATIENT_ID.isin(Xx.PATIENT_ID)]
        
            withevent=preds.loc[(preds.should_be_selected_==1)] # true positives + false negatives
            selected=preds.loc[(preds[intervention]==1)] #true positives + false positives
            benefited=preds.loc[(preds[intervention]==1) & (preds.should_be_selected_==1)] #true positives
            PPV=len(benefited)/len(selected)
            df_percentages=pd.DataFrame()
            
            df_percentages.loc['Among people with the event','N_Female']=withevent.FEMALE.sum()
            df_percentages.loc['Selected by the model','N_Female']=selected.FEMALE.sum()
            df_percentages.loc['Benefited by the intervention','N_Female']=benefited.FEMALE.sum()
            
            df_percentages['N_Male']=(pd.Series([len(withevent),len(selected),len(benefited)])-pd.Series(df_percentages['N_Female'].values)).values
            
            df_percentages.loc['PPV']=df_percentages.loc['Benefited by the intervention']/df_percentages.loc['Selected by the model']
            
            df_percentages['Female']=100*(pd.Series(df_percentages['N_Female'].values)/pd.Series([len(withevent),len(selected),len(benefited)])).values
            df_percentages['Male']=100*(pd.Series(df_percentages['N_Male'].values)/pd.Series([len(withevent),len(selected),len(benefited)])).values
            
            df_percentages.loc['Proportion avoided']=(df_percentages.loc['Benefited by the intervention']/df_percentages.loc['Among people with the event'].values)
            
            ax=df_percentages.loc[~ df_percentages.index.isin(['Proportion avoided', 'PPV']),
                                                            ['Female','Male']].plot.barh(stacked=True, title=f'{col} - {modelo} model')
            plt.axvline(df_percentages.loc['Among people with the event','Female'])
            
            numbers=[ df_percentages.loc['Among people with the event','N_Female'],
                     df_percentages.loc['Selected by the model','N_Female'],
                     df_percentages.loc['Benefited by the intervention','N_Female'],
                     df_percentages.loc['Among people with the event','N_Male'],
                    df_percentages.loc['Selected by the model','N_Male'],
                    df_percentages.loc['Benefited by the intervention','N_Male']]
            
            for p, n  in zip(ax.patches, numbers):
                width, height = p.get_width(), p.get_height(), 
                x, y = p.get_xy() 
                ax.text(x+width/2, 
                        y+height/2, 
                        ' {:.0f} ({:.0f} %)'.format(n,width), 
                        horizontalalignment='center', 
                        verticalalignment='center')
            plt.tight_layout()
            plt.savefig(figurepath+f'prop_equitativas_logistic_{modelo}_{col}_{intervention}.jpeg',dpi=300)
            
            df_percentages['AGE']= col 
            df_percentages.AGE=df_percentages.AGE.str.replace('AGE_','')
            ages=[]
            for age in df_percentages.AGE.values:
                if not age.startswith(('younger','older')): age= age[:2]+'-'+age[2:]
                ages.append(age)
            df_percentages.AGE=ages                
            df_percentages_full=pd.concat([df_percentages_full,df_percentages])
        
        print(df_percentages_full.to_latex())
        avoided=pd.DataFrame(df_percentages_full.loc['Proportion avoided','N_Female']/df_percentages_full.loc['Proportion avoided','N_Male'] )
        avoided['Male']=pd.DataFrame(df_percentages_full.loc['Proportion avoided','N_Male'] )
        avoided['Female']=pd.DataFrame(df_percentages_full.loc['Proportion avoided','N_Female'] )
        avoided['N_Male']=pd.DataFrame(df_percentages_full.loc['Benefited by the intervention','N_Male'] ).values
        avoided['N_Female']=pd.DataFrame(df_percentages_full.loc['Benefited by the intervention','N_Female'] ).values
        
        avoided['N']=(df_percentages_full.loc['Benefited by the intervention','N_Female']+df_percentages_full.loc['Benefited by the intervention','N_Male']).values
        avoided['AGE']=df_percentages_full.AGE.unique()
        avoided.index=avoided.AGE.str.replace('AGE_','')
        avoided=avoided.loc[['younger_than_54','55-64',
                                              '65-69','70-74','75-79',
                                              '80-84','older_than_85']]
    
        avoided[['Female','Male']].plot(rot=90,ls=ls,ax=ax3,color=colors[:2])
        (avoided['Male']/avoided['Female']).to_frame(name=f'Ratio M/F {modelo} model').plot(rot=90,logy=True,markersize=10,fontsize=20,ls=ls, color=colors[2], ax=ax4)
        ax4.axhline(1,color='lightgreen')
        ax3.scatter(range(len(avoided)),avoided['Female'].values,sizes=0.3*avoided.N_Female.values,color=colors[0])
        ax3.scatter(range(len(avoided)),avoided['Male'].values,sizes=0.3*avoided.N_Male.values,color=colors[1])
        ax3.set_title(f'Proportion of avoided hospitalizations')
        ax4.set_title(f'Ratio of the proportion of avoided hospitalizations')
        ax4.set_yticks([0.6,1,1.25,1.5,2])
        ax4.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        
        PPV=pd.DataFrame()
        PPV['Male']=pd.DataFrame(df_percentages_full.loc['PPV','N_Male'] )
        PPV['Female']=pd.DataFrame(df_percentages_full.loc['PPV','N_Female'] )
        PPV['N_Female']=df_percentages_full.loc['Benefited by the intervention','N_Female'].values
        PPV['N_Male']=df_percentages_full.loc['Benefited by the intervention','N_Male'].values
        PPV['AGE']=df_percentages_full.AGE.unique()
        PPV.index=PPV.AGE.str.replace('AGE_','')
        PPV=PPV.loc[['younger_than_54','55-64',
                                              '65-69','70-74','75-79',
                                              '80-84','older_than_85']]
    
        PPV[['Female','Male']].plot(rot=90,ls=ls,ax=axppv,color=colors[:2])
        (PPV['Male']/PPV['Female']).to_frame(name=f'Ratio M/F {modelo} model').plot(rot=90,logy=True,markersize=10,fontsize=20,ls=ls, color=colors[2], ax=axppv2)
        axppv2.axhline(1,color='lightgreen')
        axppv.scatter(range(len(PPV)),PPV['Female'].values,sizes=0.3*PPV.N_Female.values,color=colors[0])
        axppv.scatter(range(len(PPV)),PPV['Male'].values,sizes=0.3*PPV.N_Male.values,color=colors[1])
        axppv.set_title('Positive Predictive Values (PPV)')
        axppv2.set_title('Ratio of the PPVs')
        axppv2.set_yticks(np.logspace(np.log10((PPV['Male']/PPV['Female']).min()),np.log10((PPV['Male']/PPV['Female']).max()),4))
        for axis in [ax4.yaxis,axppv2.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_minor_formatter(ticker.NullFormatter())
        # axppv2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colorss[0][0], lw=4),
                    Line2D([0], [0], color=colorss[0][1], lw=4),
                    # Line2D([0], [0], color=colorss[1][0], lw=4),
                    # Line2D([0], [0], color=colorss[1][1], lw=4)
                    ]
    
    ax3.legend(custom_lines, ['Female', 'Male',
                              # 'Female', 'Male'
                              ],fontsize=20)
    custom_lines = [Line2D([0], [0], color=colorss[0][2], lw=4),
                    Line2D([0], [0], color=colorss[1][2], lw=4),]
    
    ax4.legend(custom_lines, ['Ratio M/F ',
                              # 'Ratio M/F  10k-10k'
                              ],
               fontsize=20,title='Scale: log10',title_fontsize=20)
    # ax4.legend(fontsize=20,title='Scale: log10',title_fontsize=20)
    # for ax in (ax33,ax34):
    ax3.title.set_fontsize(20)
    ax3.xaxis.label.set_fontsize(20)
    ax3.xaxis.set_tick_params(labelsize=20)
    ax3.yaxis.set_tick_params(labelsize=20)
    
    ax4.title.set_fontsize(20)
    ax4.xaxis.label.set_fontsize(20)
    ax4.xaxis.set_tick_params(labelsize=20)
    ax4.yaxis.set_tick_params(labelsize=20)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colorss[0][0], lw=4),
                    Line2D([0], [0], color=colorss[0][1], lw=4),
                    # Line2D([0], [0], color=colorss[1][0], lw=4),
                    # Line2D([0], [0], color=colorss[1][1], lw=4)
                    ]
    
    axppv.legend(custom_lines, ['Female', 'Male',
                              # 'Female 10k-10k', 'Male 10k-10k'
                              ],
                 fontsize=20)
    custom_lines = [Line2D([0], [0], color=colorss[0][2], lw=4),
                    Line2D([0], [0], color=colorss[1][2], lw=4),]
    
    axppv2.legend(custom_lines, ['Ratio M/F',
                              # 'Ratio M/F'
                              ],
               fontsize=20,title='Scale: log10',title_fontsize=20)
    # axppv2.legend(fontsize=20,title='Scale: log10',title_fontsize=20)
    # for ax in (axppv3,axppv4):
    axppv.title.set_fontsize(20)
    axppv.xaxis.label.set_fontsize(20)
    axppv.xaxis.set_tick_params(labelsize=20)
    axppv.yaxis.set_tick_params(labelsize=20)
    
    axppv2.title.set_fontsize(20)
    axppv2.xaxis.label.set_fontsize(20)
    axppv2.xaxis.set_tick_params(labelsize=20)
    axppv2.yaxis.set_tick_params(labelsize=20)
    
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(figurepath+f'avoided_hospit_{intervention}.jpeg',dpi=300)
    fig2.savefig(figurepath+f'PPV_{intervention}.jpeg',dpi=300)
#%%
# La figura global (la que decía Unai)
withevent=allpreds.loc[(allpreds.should_be_selected_top20k==1)]
selected=allpreds.loc[(allpreds.top20k==1)]
benefited=allpreds.loc[(allpreds.top20k==1) & (allpreds.should_be_selected_top20k==1)]
df_percentages=pd.DataFrame()

df_percentages.loc['Among people with the event','N_Female']=withevent.FEMALE.sum()
df_percentages.loc['Selected by the model','N_Female']=selected.FEMALE.sum()
df_percentages.loc['Benefited by the intervention','N_Female']=benefited.FEMALE.sum()

df_percentages['N_Male']=(pd.Series([len(withevent),len(selected),len(benefited)])-pd.Series(df_percentages['N_Female'].values)).values

df_percentages['Female']=100*(pd.Series(df_percentages['N_Female'].values)/pd.Series([len(withevent),len(selected),len(benefited)])).values
df_percentages['Male']=100*(pd.Series(df_percentages['N_Male'].values)/pd.Series([len(withevent),len(selected),len(benefited)])).values

df_percentages[['Female','Male']].plot.barh(stacked=True)
plt.axvline(df_percentages.loc['Among people with the event','Female'])
 