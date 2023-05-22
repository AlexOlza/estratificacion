#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:49:19 2023

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster


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
util.makeAllPaths()

from dataManipulation.dataPreparation import getData
import pandas as pd
import re
import numpy as np
import joblib as job
import matplotlib.pyplot as plt
from modelEvaluation.predict import predict
#%%
year=2016#eval(input('Year: '))
X, y = getData(year)
genitoCCS=X.filter(regex='PATIENT_ID|PHARMA_Benign_prostatic_hyperplasia|CCS(2[4-9]$|3[0-1]$|46$|16[3-9]$|17[0-9]$|18[0-9]$|19[0-6]$|215$)',axis=1)
patients_to_exclude=X.loc[genitoCCS.filter(regex='CCS|PHARMA').sum(axis=1)>=1]
percentwomen=100*patients_to_exclude.FEMALE.sum()/len(patients_to_exclude)
print(f'We exclude {len(patients_to_exclude)} patients with genitourinary conditions, {percentwomen} % women')

Xgen=X.loc[X.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)]#.drop(genitoCCS.filter(regex='CCS|PHARMA').columns,axis=1)
ygen=y.loc[y.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)]

Xnogen=X.loc[~X.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)].drop(genitoCCS.filter(regex='CCS|PHARMA').columns,axis=1)
ynogen=y.loc[~y.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)]

maleCancer=Xgen.loc[Xgen.FEMALE==0].loc[Xgen.filter(regex='CCS(29|30|31)').sum(axis=1)>=1]
femaleCancer=Xgen.loc[Xgen.FEMALE==1].loc[Xgen.filter(regex='CCS(25|26|27|28)').sum(axis=1)>=1]
#%%
def prevalence(y): return np.where(y[config.COLUMNS]>=1,1,0).sum()/len(y)
def percentwomen(X): return X.FEMALE.sum()/len(X)
def prev_women(X,y): return prevalence(y.loc[y.PATIENT_ID.isin(X.loc[X.FEMALE==1].PATIENT_ID)])
def prev_men(X,y): return prevalence(y.loc[y.PATIENT_ID.isin(X.loc[X.FEMALE==0].PATIENT_ID)])
def mean_conditions(X): return round(X.drop(X.filter(regex='PATIENT_ID|FEMALE|AGE').columns, axis=1).sum(axis=1).mean(),2)
def mean_cond_women(X): return mean_conditions(X.loc[X.FEMALE==1])
def mean_cond_men(X): return mean_conditions(X.loc[X.FEMALE==0])
def median_conditions(X): return round(X.drop(X.filter(regex='PATIENT_ID|FEMALE|AGE').columns, axis=1).sum(axis=1).median(),2)
def median_cond_women(X): return median_conditions(X.loc[X.FEMALE==1])
def median_cond_men(X): return median_conditions(X.loc[X.FEMALE==0])
df=pd.DataFrame([[prevalence(y),prevalence(ynogen),prevalence(ygen)],
                 [percentwomen(X),percentwomen(Xnogen),percentwomen(Xgen)],
                 [prev_women(X, y),prev_women(Xnogen, ynogen),prev_women(Xgen, ygen)],
                 [prev_men(X, y),prev_men(Xnogen, ynogen),prev_men(Xgen, ygen)],
                 [mean_conditions(X),mean_conditions(Xnogen),mean_conditions(Xgen)],
                 [mean_cond_women(X),mean_cond_women(Xnogen),mean_cond_women(Xgen)],
                 [mean_cond_men(X),mean_cond_men(Xnogen),mean_cond_men(Xgen)]])
df=df.rename(index={0:'prevalence',1:'percent_women',2:'prev_women',
                      3:'prev_men',4:'mean_conditions',5:'mean_cond_women',
                      6:'mean_cond_men'},
             columns={0:'global',1:'no_gen',2:'gen'})

print(df.round(2).to_markdown())
prev_men(maleCancer,ygen)
prev_women(femaleCancer,ygen)
#%%
cols=[c for c in Xgen if ((not c in genitoCCS.columns) or (c=='PATIENT_ID'))]
df_genito=pd.DataFrame([[mean_conditions(Xgen[list(genitoCCS.columns)+list(['FEMALE'])]),
                         mean_conditions(Xgen[cols])],
                        [mean_cond_women(Xgen[list(genitoCCS.columns)+list(['FEMALE'])]),
                         mean_cond_women(Xgen[cols])],
                        [mean_cond_men(Xgen[list(genitoCCS.columns)+list(['FEMALE'])]),
                         mean_cond_men(Xgen[cols])]],
                       index=['genito','genito_women','genito_men'],
                       columns=['mean_genito_cond','mean_other_cond'])
print(df_genito.round(2).to_markdown())
#%%
model_drop_columns=job.load(config.ROOTPATH+'/models/urgcmsCCS_non_genitourinary/logistic20230201_161315.joblib')
model_keep_columns=job.load(config.ROOTPATH+'/models/urgcmsCCS_parsimonious/logistic20230201_121805.joblib')
model_drop_patients=job.load(config.ROOTPATH+'/models/urgcmsCCS_vigo/logistic20230213_104105.joblib')

def coef(model): return {k:[v] for k,v in zip(model.feature_names_in_, model.coef_[0])}

df_coef=pd.DataFrame([[coef(model_keep_columns)['FEMALE'][0],
                       coef(model_drop_columns)['FEMALE'][0],
                       coef(model_drop_patients)['FEMALE'][0]],
                     [model_keep_columns.intercept_[0],model_drop_columns.intercept_[0],
                      model_drop_patients.intercept_[0]]],
                     columns=['keep_cols','drop_cols','drop_pat'],
                     index=['beta_FEMALE','intercept']).T
df_coef['p_healthyFemale']=np.exp(df_coef.intercept+df_coef.beta_FEMALE)/(1+np.exp(df_coef.intercept+df_coef.beta_FEMALE))
df_coef['p_healthyMale']=np.exp(df_coef.intercept)/(1+np.exp(df_coef.intercept))
print(df_coef.round(3).to_markdown())

#%%
X17,y17=getData(2017)
genitoCCS=X17.filter(regex='PATIENT_ID|PHARMA_Benign_prostatic_hyperplasia|CCS(2[4-9]$|3[0-1]$|46$|16[3-9]$|17[0-9]$|18[0-9]$|19[0-6]$|215$)',axis=1)
patients_to_exclude=X17.loc[genitoCCS.filter(regex='CCS|PHARMA').sum(axis=1)>=1]
X17nogen=X17.loc[~X17.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)].drop(genitoCCS.filter(regex='CCS|PHARMA').columns,axis=1)
y17nogen=y17.loc[~y17.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)]

#%%
preds_drop_columns,_=predict('logistic20230201_161315', 'urgcmsCCS_non_genitourinary', 2018,
                           X=X17.drop(genitoCCS.drop('PATIENT_ID',axis=1).columns,axis=1), y=y17,
                           pastX=X.drop(genitoCCS.drop('PATIENT_ID',axis=1).columns,axis=1), pasty=y)
preds_keep_columns,_=predict('logistic20230201_121805', 'urgcmsCCS_parsimonious', 2018, X=X17,y=y17,pastX=X,pasty=y)

preds_drop_patients,_=predict('logistic20230213_104105','urgcmsCCS_vigo',2018,X=X17nogen,y=y17nogen,pastX=Xnogen,pasty=ynogen)
#%%
import seaborn as sns
fig,ax=plt.subplots()
sns.set_style('whitegrid')
plt.xlim(0, 0.2)
sns.kdeplot(preds_drop_columns.PRED,ax=ax,label='General Population')
sns.kdeplot(preds_drop_columns.loc[preds_drop_columns.PATIENT_ID.isin(patients_to_exclude.PATIENT_ID)].PRED,ax=ax, label='Excluded patients')
plt.legend()
#%%
""" AGE DISTRIBUTION """
def age_distr(X):
    if not 'AGE_85+' in X:
        X['AGE_85+']=np.where(X.filter(regex=("AGE*")).sum(axis=1)==0,1,0)    
    ages=pd.DataFrame(X.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100,
                      columns=['Porcentaje'])
    X.drop('AGE_85+',axis=1,inplace=True)
    return ages
def new_age_groups(agesRisk):
    agesRisk['Grupo edad']=agesRisk.index
    agesRisk['Grupo edad']=agesRisk['Grupo edad'].str.replace('AGE_','')
    agesRisk['Grupo edad']=np.where(agesRisk['Grupo edad'].str.contains('85+'),
                                    agesRisk['Grupo edad'],
                                    agesRisk['Grupo edad'].str[:2]+'-'+agesRisk['Grupo edad'].str[2:])
    agesRisk=agesRisk[['Grupo edad', 'Porcentaje']]
    agesRisk.loc['AGE_7584']=['75-84',agesRisk.loc['AGE_7579'].Porcentaje+agesRisk.loc['AGE_8084'].Porcentaje]
    agesRisk.loc['AGE_6574']=['65-74',agesRisk.loc['AGE_6569'].Porcentaje+agesRisk.loc['AGE_7074'].Porcentaje]
    agesRisk.loc['AGE_0018']=['0-18', agesRisk.loc['AGE_0004'].Porcentaje+agesRisk.loc['AGE_0511'].Porcentaje+agesRisk.loc['AGE_1217'].Porcentaje]
    agesRisk.loc['AGE_1854']=['18-54',agesRisk.loc['AGE_1834'].Porcentaje+agesRisk.loc['AGE_3544'].Porcentaje+agesRisk.loc['AGE_4554'].Porcentaje]
    agesRisk=agesRisk.loc[['AGE_0018','AGE_1854','AGE_5564','AGE_6574','AGE_7584','AGE_85+']]
    return agesRisk
def plot_ages(X, title):
    ages=new_age_groups(age_distr(X))
    labels = [f'{group}' for group in ages.index]
    plt.pie(ages.Porcentaje.values,labels=labels,autopct='%.2f')
    plt.axis('equal')
    plt.title(title)
    plt.show()
    
plot_ages(Xgen.loc[Xgen.FEMALE==1], 'Genitourinary female patients')
plot_ages(Xgen.loc[Xgen.FEMALE==0], 'Genitourinary male patients')

plot_ages(Xnogen.loc[Xnogen.FEMALE==1], 'Non-genitourinary female patients')
plot_ages(Xnogen.loc[Xnogen.FEMALE==0], 'Non-genitourinary male patients')

plot_ages(X.loc[X.FEMALE==1], 'All female patients')
plot_ages(X.loc[X.FEMALE==0], 'All male patients')