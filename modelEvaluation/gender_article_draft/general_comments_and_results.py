#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:28:50 2023

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
from matplotlib import pyplot as plt
figurepath='/home/aolza/Desktop/estratificacion/figures/gender_article_draft'

logistic_modelpath=config.ROOTPATH+'models/urgcmsCCS_parsimonious/'
linear_modelpath=config.ROOTPATH+'models/costCCS_parsimonious/'

logistic_modelname='logistic20230324_111354'
linear_modelname='linear20230324_130625'

logistic_predpath=re.sub('models','predictions',logistic_modelpath)
linear_predpath=re.sub('models','predictions',linear_modelpath)

logistic_global_model=joblib.load(logistic_modelpath+f'{logistic_modelname}.joblib')
linear_global_model=joblib.load(linear_modelpath+f'{linear_modelname}.joblib')
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
def concat_preds(file1,file2):
    muj=pd.read_csv(file1)
    muj['FEMALE']=1
    hom=pd.read_csv(file2)
    hom['FEMALE']=0
    return pd.concat([muj,hom])
for path,modeltype in zip([linear_predpath,logistic_predpath],['linear','logistic']):
    cal='calibrated' if modeltype=='logistic' else ''
    separate_preds_w=pd.read_csv(path+f'{modeltype}Mujeres_{cal}_2018.csv')
    separate_preds_m=pd.read_csv(path+f'{modeltype}Hombres_{cal}_2018.csv')  
    separate_preds_m['Gender']='Male' ; separate_preds_w['Gender']='Female'
    if modeltype=='linear':
        separate_preds_m.loc[separate_preds_m.PRED<0,'PRED']=0
        separate_preds_w.loc[separate_preds_w.PRED<0,'PRED']=0
    separate_preds=pd.concat([separate_preds_m,separate_preds_w]).reset_index()
    df=pd.concat([separate_preds_m.PRED.describe().rename('Men'),separate_preds_w.PRED.describe().rename('Women')],axis=1)
    print(df)
    
    data=separate_preds.nlargest(20000,'PRED')
    if modeltype=='linear':
        data=data.nsmallest(len(data)-2,'PRED') #remove two outliers
        xlabel='Predicted cost in euros'
    else:
        xlabel='Predicted probability'

    fig,ax=plt.subplots()
    plot=sns.kdeplot(data=data,x='PRED',hue='Gender', ax=ax,clip=(data.PRED.min(),data.PRED.max()))
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(figurepath+f'/{modeltype}_separate_predictions_density.jpeg', dpi=300)
#%%

"""
HOSPITALIZATION BIG TABLE
"""

def patient_selection(x,modeltype):
    x['top20k']=np.where(x.PATIENT_ID.isin(x.nlargest(20000,'PRED').PATIENT_ID),1,0)
    top10k_women=x.loc[x.FEMALE==1].nlargest(10000,'PRED')
    top10k_men=x.loc[x.FEMALE==0].nlargest(10000,'PRED')
    x['top10k_gender']=np.where(x.PATIENT_ID.isin(pd.concat([top10k_women,top10k_men]).PATIENT_ID),1,0)
    if modeltype=='linear':
        x['should_be_selected']=np.where(x.PATIENT_ID.isin(x.nlargest(20000,'OBS').PATIENT_ID),1,0)
    else:
        x['should_be_selected']=np.where(x.OBS>=1,1,0)
    return x
from sklearn.metrics import confusion_matrix
def cm(x,col): return {key: val for key, val in zip(['tn', 'fp', 'fn', 'tp'],confusion_matrix(x[f'should_be_selected'],x[col]).ravel())} 
def threshold_muj_hom(x,col):
    c=cm(x.loc[x.FEMALE==1],col)
    vpp_women, vpn_women, sens_women, esp_women=c['tp']/(c['tp']+c['fp']),    c['tn']/(c['tn']+c['fn']),    c['tp']/(c['tp']+c['fn']),    c['tn']/(c['tn']+c['fp'])
    c=cm(x.loc[x.FEMALE==0],col)
    vpp_men, vpn_men, sens_men, esp_men=c['tp']/(c['tp']+c['fp']),    c['tn']/(c['tn']+c['fn']),    c['tp']/(c['tp']+c['fn']),    c['tn']/(c['tn']+c['fp'])
    return(vpp_women, vpp_men, vpn_women, vpn_men, sens_women, sens_men, esp_women, esp_men)

import numpy as np
def table(path, modelname):
    modeltype=re.sub('[0-9]|_','',modelname) #this string will contain either "logistic" or "linear"
    cal='calibrated' if modeltype=='logistic' else ''
    # Global model
    global_preds=concat_preds(path+f'{modelname}_Mujeres_{cal}_2018.csv',
                              path+f'{modelname}_Hombres_{cal}_2018.csv')   
    # Separate models 
    separate_preds=concat_preds(path+f'{modeltype}Mujeres_{cal}_2018.csv',
                                path+f'{modeltype}Hombres_{cal}_2018.csv')    
    problematic=((separate_preds.PRED-separate_preds.OBS)**2).nlargest(10).index
    separate_unproblematic_preds=separate_preds.loc[~ separate_preds.index.isin(problematic)] if modeltype=='linear' else separate_preds
    if modeltype=='logistic':
        index=['global','separate','same_prevalence']
        # Same prevalence model
        sameprevalence_preds=concat_preds(path+'logistic_gender_balanced_Mujeres_calibrated_2018.csv',
                                          path+'logistic_gender_balanced_Hombres_calibrated_2018.csv')
        allpreds=[global_preds,separate_unproblematic_preds,sameprevalence_preds]
        # Overall metrics: auc and brier
        from sklearn.metrics import roc_auc_score,brier_score_loss 
        def AUC_muj(x): return roc_auc_score(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def AUC_hom(x): return roc_auc_score(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        def brier_muj(x): return brier_score_loss(np.where(x.loc[x.FEMALE==1].OBS,1,0),x.loc[x.FEMALE==1].PRED)
        def brier_hom(x): return brier_score_loss(np.where(x.loc[x.FEMALE==0].OBS,1,0),x.loc[x.FEMALE==0].PRED)
        
        overall_metrics=[[AUC_muj(x),AUC_hom(x),brier_muj(x),brier_hom(x)] for x in allpreds]
        df_overall=pd.DataFrame(overall_metrics,columns=['AUC_women','AUC_men','brier_women','brier_men'],index=index)
    else:
        index=['global','separate']
        allpreds=[global_preds,separate_unproblematic_preds]
        #Overall metrics: R2 and RMSE
        from sklearn.metrics import r2_score,mean_squared_error 
        def R2_muj(x): return r2_score(x.loc[x.FEMALE==1].OBS,x.loc[x.FEMALE==1].PRED)
        def R2_hom(x): return r2_score(x.loc[x.FEMALE==0].OBS,x.loc[x.FEMALE==0].PRED)
        def RSS_muj(x): return ((x.loc[x.FEMALE==1].OBS-x.loc[x.FEMALE==1].PRED)**2).sum()
        def RSS_hom(x): return ((x.loc[x.FEMALE==0].OBS-x.loc[x.FEMALE==0].PRED)**2).sum()
        def RMSE_muj(x): return mean_squared_error(x.loc[x.FEMALE==1].OBS,x.loc[x.FEMALE==1].PRED,squared=False)
        def RMSE_hom(x): return mean_squared_error(x.loc[x.FEMALE==0].OBS,x.loc[x.FEMALE==0].PRED,squared=False)
        
        overall_metrics=[[R2_muj(x),R2_hom(x),RMSE_muj(x),RMSE_hom(x),RSS_muj(x),RSS_hom(x)] for x in allpreds]
        df_overall=pd.DataFrame(overall_metrics,columns=['R2_women','R2_men','RMSE_women','RMSE_men','RSS_women','RSS_men'],index=index)

    # PATIENT SELECTION:
    # We use two criteria: 
        # 1) Select the top 20k patients as positive, regardless of gender -> top20k
        # 2) Select the top 10k women and top 10k men -> top10k_gender
    
    global_preds=patient_selection(global_preds,modeltype)
    separate_unproblematic_preds=patient_selection(separate_unproblematic_preds,modeltype)
    if modeltype=='logistic':
        sameprevalence_preds=patient_selection(sameprevalence_preds,modeltype)
    
    
    # In either case, we use the same threshold-specific metrics
    threshold_metrics=[np.array([threshold_muj_hom(x, 'top20k'),threshold_muj_hom(x, 'top10k_gender')]).ravel() for  x in allpreds]
    
    threshold_metrics=[list(e)+list([100*x.loc[x.top20k==1].FEMALE.sum()/x.top20k.sum()]) for e,x in zip(threshold_metrics, allpreds)]
    
    df_threshold=pd.DataFrame(threshold_metrics,columns=['PPV_20k_women','PPV_20k_men','NPV_20k_women','NPV_20k_men','SENS_20k_women','SENS_20k_men','SPEC_20k_women','SPEC_20k_men',
                                                'PPV_10k_women','PPV_10k_men','NPV_10k_women','NPV_10k_men','SENS_10k_women','SENS_10k_men','SPEC_10k_women','SPEC_10k_men',
                                                'Perc_top20k_women'],
                 index=index)
    
    df=pd.concat([df_overall,df_threshold],axis=1)
    
    nmen=len(global_preds)-global_preds.FEMALE.sum()
    print('N women ',global_preds.FEMALE.sum())
    print('N men ',nmen)
    print('Prevalence women: ',np.where(global_preds.loc[global_preds.FEMALE==1].OBS,1,0).sum()/global_preds.FEMALE.sum())
    print('Prevalence men: ',np.where(global_preds.loc[global_preds.FEMALE==0].OBS,1,0).sum()/nmen)
    
    return df
#%%
df_logistic=table(logistic_predpath,logistic_modelname)
print(df_logistic.T.round(3).to_latex())
#%%
df_log_round=df_logistic.round(3)
cols=pd.Series([re.sub('_men|_women','',c) for c in df_logistic]).drop_duplicates()
df=pd.DataFrame(index=df_log_round.index)
for col in cols:
    df[col]=[c for c in df_log_round.filter(regex=col).values]
print(df.T.to_latex())

#%%
df_linear=table(linear_predpath,linear_modelname)
print(df_linear.T.round(3).to_latex())
#%%
df_lin_round=df_linear.round(3)
cols=pd.Series([re.sub('_men|_women','',c) for c in df_linear]).drop_duplicates()
df=pd.DataFrame(index=df_lin_round.index)
for col in cols:
    df[col]=[c for c in df_lin_round.filter(regex=col).values]
print(df.T.to_latex())
