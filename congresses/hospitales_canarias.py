#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:08:20 2023

@author: aolza

Modelo predictivo de riesgo de muerte a los 12 meses como elemento de ayuda para la identificación de personas con necesidades de cuidados paliativos

| Model                              |      AUC |       AP |    R@20k |   PPV@20K |      Brier |
|:-----------------------------------|---------:|---------:|---------:|----------:|-----------:|
| paliativos_canarias_Demo           | 0.911928 | 0.097352 | 0.177472 |   0.1447  | 0.0092119  |
| paliativos_canarias_DemoDiag       | 0.946049 | 0.2234   | 0.265473 |   0.2979  | 0.0086666  |
| paliativos_canarias_DemoDiagPharma | 0.948048 | 0.237277 | 0.27728  |   0.31115 | 0.00856241 |


| Model                           |      AUC |       AP |    R@20k |   PPV@20K |      Brier |
|:--------------------------------|---------:|---------:|---------:|----------:|-----------:|
| constrained_Demo                | 0.911928 | 0.097352 | 0.177472 |   0.1447  | 0.0092119  |
| constrained_DemoDiag            | 0.944004 | 0.21197  | 0.255269 |   0.28645 | 0.00874586 |
| constrained_DemoDiagPharma      | 0.945396 | 0.219565 | 0.262309 |   0.29435 | 0.00868981 |
| constrained_DemoDiag_freePharma | 0.946432 | 0.22596  | 0.267968 |   0.3007  | 0.00864439 |
"""

import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.death1year_canarias'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()
import joblib
import pandas as pd
import numpy as np
from dataManipulation.dataPreparation import getData
from modelEvaluation.compare import performance
#%%
X, y = getData(2017)
#%%
descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
model=joblib.load(config.MODELPATH+'/nested_logistic/constrained_DemoDiag_freePharma.joblib')
preds=pd.read_csv(config.PREDPATH+'/nested_logistic/constrained_DemoDiag_freePharma__2018.csv')

#%%
future_dead=pd.read_csv(config.FUTUREDECEASEDFILE)
dead2019=future_dead.loc[future_dead.date_of_death.str.startswith('2019')].PATIENT_ID
preds['OBS2019']=np.where(preds.PATIENT_ID.isin(dead2019),1,0)
#%%
baseline_risk=np.exp(model.intercept_[0])
print('Probabilidad basal de fallecer: ',baseline_risk)

recall2019, ppv2019, _, _ = performance(obs=preds.OBS+preds.OBS2019, pred=preds.PRED, K=20000)
recall, ppv, spec, newpred= performance(obs=preds.OBS, pred=preds.PRED, K=20000)
preds['TopK']=newpred.astype(int)
#%%
feature_names=X.filter(regex='FEMALE|AGE_[0-9]+$|CCS|PHARMA',axis=1).columns
# coefs=pd.DataFrame.from_dict({k:[v] for v,k in zip(model.coef_[0],model.feature_names_in_)},orient='columns')
coefs=pd.DataFrame.from_dict({k:[v] for v,k in zip(model.coef_[0],feature_names)},orient='columns')

coefs.rename(columns={0:'beta'},inplace=True)
coefsT=coefs.T
coefsT['CATEGORIES']=coefsT.index
sorted_coefs=pd.merge(coefsT.sort_values(0,ascending=False),descriptions,on='CATEGORIES',how='left')
print(sorted_coefs.loc[sorted_coefs[0]!=0].to_markdown())
# coefs=coefs.T.loc[coefs.T[0]!=0].T
#%%
def table_1(preds,X,descriptions, chosen_CCSs):
    topK=X.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]
    table={}
    for chosen_CCS in chosen_CCSs:
        inlist=topK[chosen_CCS].sum()
        inpopulation=X[chosen_CCS].sum()
        table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[f'{inlist} ({round(100*inlist/len(topK),2)} %)',f'{inpopulation} ({round(100*inpopulation/len(X),2)} %)']
    table=pd.DataFrame.from_dict(table,orient='index',columns=['Listado', 'Población General'])
    print(table.to_markdown())
    return table

table1=table_1(preds,X,descriptions, chosen_CCSs=['CCS1','CCS2','CCS3'])
#%%

if not 'AGE_85+' in X:
    X['AGE_85+']=np.where(X.filter(regex=("AGE*")).sum(axis=1)==0,1,0)
    
Xx=X.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]
Yy=y.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]

agesRisk=pd.DataFrame(Xx.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100)
agesGP=pd.DataFrame(X.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100)

import seaborn as sns
import matplotlib.pyplot as plt
fig, ax=plt.subplots(figsize=(8,10))


agesRisk['Grupo edad']=agesRisk.index
agesRisk['Porcentaje']=agesRisk[0]
agesRisk=agesRisk[['Grupo edad', 'Porcentaje']]
# agesRisk.loc[len(agesRisk.index)]=['AGE_85+',a85plus]
agesGP['Grupo edad']=agesGP.index
agesGP['Porcentaje']=agesGP[0]
agesGP=agesGP[['Grupo edad', 'Porcentaje']]
# agesGP.loc[len(agesGP.index)]=['AGE_85+',a85plus]
sns.barplot( y='Grupo edad',x='Porcentaje',ax=ax,data=agesGP,
            order=sorted(agesRisk['Grupo edad'].values),color='b', alpha=0.5, label='Población general')

sns.barplot( y='Grupo edad',x='Porcentaje',ax=ax,data=agesRisk,
            order=sorted(agesRisk['Grupo edad'].values),color='r', alpha=0.5,label='Grupo riesgo')
ax.legend()
#%%
def explain_example(lower,upper, preds, coefs, X, descriptions, random_state):
    patient_A=preds.loc[preds.PRED>lower].loc[preds.PRED<upper].sample(1,random_state=random_state)
    patient_A_CCS=X.loc[X.PATIENT_ID==patient_A.PATIENT_ID.values[0]]
    patient_A_CCS=[c for c in patient_A_CCS if patient_A_CCS[c].values[0]==1]
    smallest=coefs[patient_A_CCS].T.nsmallest(5,0)
    df_A=pd.concat([coefs[patient_A_CCS].T.nlargest(5,0),smallest.loc[smallest[0]<0]])
    df_A['CATEGORIES']=df_A.index
    df_A=pd.merge(df_A,descriptions,on='CATEGORIES',how='left')
    return patient_A, df_A

patient_A, df_A=explain_example(0.95,0.96,preds.loc[preds.TopK==1],coefs,X,descriptions,0)
patient_B, df_B=explain_example(0.2,0.25,preds.loc[preds.TopK==1],coefs,X,descriptions,0)
#%%
