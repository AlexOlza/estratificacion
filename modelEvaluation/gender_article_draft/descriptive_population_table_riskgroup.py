#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:32:55 2023

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

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
    return agesRisk
def table_1(X,y,Xcost, ycost, descriptions, chosen_CCSs,ages):
    table={}
    men=X.loc[X.FEMALE==0]
    women=X.loc[X.FEMALE==1]
    for chosen_CCS_group in chosen_CCSs:
        if isinstance(chosen_CCS_group, int):
            chosen_CCS=f'CCS{chosen_CCS_group}'
            inpopulation=X[chosen_CCS].sum()
            inmen=men[chosen_CCS].sum()
            inwomen=women[chosen_CCS].sum()
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inpopulation/len(X),2),round(100*inmen/len(men),2) ,round(100*inwomen/len(women),2) ]
            
            #f'{inmen} ({round(100*inmen/len(topK),2)} %)',f'{inwomen} ({round(100*inwomen/len(X),2)} %)']
        else:
            men_=men.copy()
            women_=women.copy()
            X_=X.copy()
            inmen,inwomen, inpopulation= 0, 0, 0
            for ccs in chosen_CCS_group:
                chosen_CCS=f'CCS{ccs}'
                print(X_[chosen_CCS].sum(),'people have ',chosen_CCS)
                inpopulation+=X_[chosen_CCS].sum()
                inmen+=men_[chosen_CCS].sum()
                inwomen+=women_[chosen_CCS].sum()
                men_=men_.loc[men_[chosen_CCS]==0]
                women_=women_.loc[women_[chosen_CCS]==0]
                X_=X_.loc[X_[chosen_CCS]==0]
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inpopulation/len(X),2),round(100*inmen/len(men),2) ,round(100*inwomen/len(women),2) ]        
    table=pd.DataFrame.from_dict(table,orient='index',columns=['All','Male', 'Female'])
    # print(table.sort_values('Mujeres',ascending=False).to_markdown())
    agesRisk_men=new_age_groups(men)
    agesRisk_women=new_age_groups(women)
    agesRisk_population=new_age_groups(X)
    print(agesRisk_men)
    table.loc['Older than 65']=[round(agesRisk_population.loc[[ 'AGE_6574', 'AGE_7584', 'AGE_85+']].Porcentaje.sum(),2),
                                round(agesRisk_men.loc[[ 'AGE_6574', 'AGE_7584', 'AGE_85+']].Porcentaje.sum(),2),
                                round(agesRisk_women.loc[[ 'AGE_6574', 'AGE_7584', 'AGE_85+']].Porcentaje.sum(),2)]
    for col in ages:
        table.loc[col]=[round(agesRisk_population.loc[col].Porcentaje,2),
                                    round(agesRisk_men.loc[col].Porcentaje,2),
                                    round(agesRisk_women.loc[col].Porcentaje,2)]
        
    y['FEMALE']=X.FEMALE
    y['hospit']=np.where(y.urgcms>=1,1,0)
    prev_all=y.hospit.sum()*100/len(y)
    prev_men=y.loc[y.FEMALE==0].hospit.sum()*100/len(y.loc[y.FEMALE==0])
    prev_women=y.loc[y.FEMALE==1].hospit.sum()*100/len(y.loc[y.FEMALE==1])
    table.loc['Prevalence of hospitalization']=[round(prev_all,2),round(prev_men,2),round(prev_women,2)]
    
    ycost['FEMALE']=Xcost.FEMALE
    mean_all=ycost.COSTE_TOTAL_ANO2.mean()
    mean_men=ycost.loc[ycost.FEMALE==0].COSTE_TOTAL_ANO2.mean()
    mean_women=ycost.loc[ycost.FEMALE==1].COSTE_TOTAL_ANO2.mean()
    table.loc['Mean cost (euro)']=[round(mean_all,2),round(mean_men,2),round(mean_women,2)]
    
    
    return table

def concat_preds(file1,file2):
    muj=pd.read_csv(file1)
    muj['FEMALE']=1
    hom=pd.read_csv(file2)
    hom['FEMALE']=0
    return pd.concat([muj,hom])
#%%
if __name__=='__main__':
    X, y = getData(2017, columns=['urgcms'])
    Xcost, ycost=getData(2017, columns=['COSTE_TOTAL_ANO2'])
    #%%
    descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
    
    #%%
    feature_names=X.filter(regex='FEMALE|AGE_[0-9]+$|CCS|PHARMA',axis=1).columns
    logistic_modelpath=config.ROOTPATH+'models/urgcmsCCS_parsimonious/'
    logistic_predpath=config.ROOTPATH+'predictions/urgcmsCCS_parsimonious/'
    logistic_modelname='logistic20230324_111354'
    model=joblib.load(logistic_modelpath+f'{logistic_modelname}.joblib')
    preds=concat_preds(logistic_predpath+f'{logistic_modelname}_Mujeres_calibrated_2018.csv',
                              logistic_predpath+f'{logistic_modelname}_Hombres_calibrated_2018.csv')  
    
    #%%
    recall, ppv, spec, newpred= performance(obs=preds.OBS, pred=preds.PRED, K=20000)
    preds['TopK']=newpred.astype(int)
    #%%
    chosen_CCSs=[[i for i in range(11,46)],[98,99],127,128,[49,50],108,158,79,202,659,83,206,205,657,[100,101]]
    coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(model.feature_names_in_, model.coef_[0])},
                                            orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()
    large_betas=[int(c[3:]) for c in coefs.nlargest(5,'beta').CATEGORIES]
    small_betas=[int(c[3:]) for c in coefs.nsmallest(15,'beta').CATEGORIES if not c[:3]=='AGE']
    for c in list(large_betas)+list(small_betas):
        chosen_CCSs.append(c)
    #%%
    ages=[c for c in X.columns if 'AGE' in c]
    t1=table_1(X,y,Xcost, ycost, descriptions, chosen_CCSs,ages)
    
    Xtopk=X.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]
    Xcosttopk=Xcost.loc[Xcost.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]
    print(t1.to_latex())
    
    t1bis=table_1(Xtopk, y.loc[y.PATIENT_ID.isin(Xtopk.PATIENT_ID)],
                  Xcosttopk, ycost.loc[ycost.PATIENT_ID.isin(Xcosttopk.PATIENT_ID)],
                  descriptions, chosen_CCSs,ages)
    print(t1bis.to_latex())    
    
    fullt1=pd.merge(t1,t1bis,left_index=True,right_index=True)
