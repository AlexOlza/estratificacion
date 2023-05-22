#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:32:55 2023

@author: alex
"""
import sys
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcmsCCS_parsimonious'
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
Xcost, ycost=getData(2017, columns='COSTE_TOTAL_ANO2')
#%%
descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')

#%%
feature_names=X.filter(regex='FEMALE|AGE_[0-9]+$|CCS|PHARMA',axis=1).columns
#%%
def new_age_groups(X):
    if not 'AGE_85+' in X:
        X['AGE_85+']=np.where(X.filter(regex=("AGE*")).sum(axis=1)==0,1,0)
    
    agesRisk=pd.DataFrame(X.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100)
    agesRisk['Grupo edad']=agesRisk.index
    agesRisk['Grupo edad']=agesRisk['Grupo edad'].str.replace('AGE_','')
    agesRisk['Grupo edad']=np.where(agesRisk['Grupo edad'].str.contains('85+'),
                                    agesRisk['Grupo edad'],
                                    agesRisk['Grupo edad'].str[:2]+'-'+agesRisk['Grupo edad'].str[2:])
    agesRisk['Porcentaje']=agesRisk[0]
    agesRisk=agesRisk[['Grupo edad', 'Porcentaje']]
    agesRisk.loc['AGE_7584']=['75-84',agesRisk.loc['AGE_7579'].Porcentaje+agesRisk.loc['AGE_8084'].Porcentaje]
    agesRisk.loc['AGE_6574']=['65-74',agesRisk.loc['AGE_6569'].Porcentaje+agesRisk.loc['AGE_7074'].Porcentaje]
    agesRisk.loc['AGE_1854']=['18-54',agesRisk.loc['AGE_1834'].Porcentaje+agesRisk.loc['AGE_3544'].Porcentaje+agesRisk.loc['AGE_4554'].Porcentaje]
    agesRisk=agesRisk.loc[['AGE_1854','AGE_5564','AGE_6574','AGE_7584','AGE_85+']]
    return agesRisk
def table_1(X,y,Xcost, ycost, descriptions, chosen_CCSs):
    table={}
    men=X.loc[X.FEMALE==0]
    women=X.loc[X.FEMALE==1]
    for chosen_CCS_group in chosen_CCSs:
        if isinstance(chosen_CCS_group, int):
            chosen_CCS=f'CCS{chosen_CCS_group}'
            inlist=men[chosen_CCS].sum()
            inpopulation=women[chosen_CCS].sum()
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inlist/len(men),2) ,round(100*inpopulation/len(women),2) ]
            
            #f'{inlist} ({round(100*inlist/len(topK),2)} %)',f'{inpopulation} ({round(100*inpopulation/len(X),2)} %)']
        else:
            inlist,inpopulation= 0, 0
            for ccs in chosen_CCS_group:
                chosen_CCS=f'CCS{ccs}'
                # print(topK[chosen_CCS].sum(),'people have ',chosen_CCS)
                inlist+=men[chosen_CCS].sum()
                inpopulation+=women[chosen_CCS].sum()
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inlist/len(men),2) ,round(100*inpopulation/len(women),2) ]        
    table=pd.DataFrame.from_dict(table,orient='index',columns=['Male', 'Female'])
    # print(table.sort_values('Mujeres',ascending=False).to_markdown())
    agesRisk_men=new_age_groups(men)
    agesRisk_women=new_age_groups(women)
    table.loc['Older than 65']=[round(agesRisk_men.loc[[ 'AGE_6574', 'AGE_7584', 'AGE_85+']].Porcentaje.sum(),2),
                                round(agesRisk_women.loc[[ 'AGE_6574', 'AGE_7584', 'AGE_85+']].Porcentaje.sum(),2)]
    
    y['FEMALE']=X.FEMALE
    y['hospit']=np.where(y.urgcms>=1,1,0)
    prev_men=y.loc[y.FEMALE==0].hospit.sum()*100/len(y.loc[y.FEMALE==0])
    prev_women=y.loc[y.FEMALE==1].hospit.sum()*100/len(y.loc[y.FEMALE==1])
    table.loc['Prevalence of hospitalization']=[round(prev_men,2),round(prev_women,2)]
    
    ycost['FEMALE']=Xcost.FEMALE
    mean_men=ycost.loc[ycost.FEMALE==0].COSTE_TOTAL_ANO2.mean()
    mean_women=ycost.loc[ycost.FEMALE==1].COSTE_TOTAL_ANO2.mean()
    table.loc['Mean cost (euro)']=[round(mean_men,2),round(mean_women,2)]
    
    
    return table

chosen_CCSs=[[i for i in range(11,46)],[98,99],127,128,[49,50],108,158,79,202,659,83,206,205,657,[100,101]]

table1=table_1(X,y, Xcost, ycost,descriptions, chosen_CCSs=chosen_CCSs)

print(table1.to_latex())

