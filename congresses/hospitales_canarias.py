#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:08:20 2023

@author: alex

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
sys.path.append('/home/alex/Desktop/estratificacion/')#necessary in cluster

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
baseline_risk=np.exp(model.intercept_[0])/(1+np.exp(model.intercept_[0]))
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
    for chosen_CCS_group in chosen_CCSs:
        if isinstance(chosen_CCS_group, int):
            chosen_CCS=f'CCS{chosen_CCS_group}'
            inlist=topK[chosen_CCS].sum()
            inpopulation=X[chosen_CCS].sum()
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inlist/len(topK),2) ,round(100*inpopulation/len(X),2) ]
            
            #f'{inlist} ({round(100*inlist/len(topK),2)} %)',f'{inpopulation} ({round(100*inpopulation/len(X),2)} %)']
        else:
            inlist,inpopulation= 0, 0
            X_=X.copy()
            topK_=topK.copy()
            for ccs in chosen_CCS_group:
                chosen_CCS=f'CCS{ccs}'
                print(topK_[chosen_CCS].sum(),'people have ',chosen_CCS)
                inlist+=topK_[chosen_CCS].sum()
                inpopulation+=X_[chosen_CCS].sum()
                X_=X_.loc[X_[chosen_CCS]==0]
                topK_=topK_.loc[topK_[chosen_CCS]==0]
            table[descriptions.loc[descriptions.CATEGORIES==chosen_CCS].LABELS.values[0]]=[round(100*inlist/len(topK),2) ,round(100*inpopulation/len(X),2) ]
    table['Mujeres']=[100*topK['FEMALE'].sum()/len(topK),100*X['FEMALE'].sum()/len(X)]
    table=pd.DataFrame.from_dict(table,orient='index',columns=['Listado', 'Población General'])
    print(table.sort_values('Listado',ascending=False).to_markdown())
    return table

chosen_CCSs=[[i for i in range(11,46)],[98,99],127,128,[49,50],108,158,79,202,659,83,206,205,657,[100,101]]

table1=table_1(preds,X,descriptions, chosen_CCSs=chosen_CCSs)

#%%

if not 'AGE_85+' in X:
    X['AGE_85+']=np.where(X.filter(regex=("AGE*")).sum(axis=1)==0,1,0)
    
Xx=X.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]
Yy=y.loc[X.PATIENT_ID.isin(preds.loc[preds.TopK==1].PATIENT_ID)]

agesRisk=pd.DataFrame(Xx.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100)
agesGP=pd.DataFrame(X.filter(regex=("AGE*")).idxmax(1).value_counts(normalize=True)*100)

import matplotlib.pyplot as plt
def new_age_groups(agesRisk):
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
# agesRisk.loc[len(agesRisk.index)]=['AGE_85+',a85plus]
agesRisk=new_age_groups(agesRisk)
agesGP=new_age_groups(agesGP)

def fmt(x):
    return '{:.1f}%'.format(x)

labels = [f'{group}' for group, perc in zip(agesRisk['Grupo edad'],agesRisk['Porcentaje'])]

plt.figure(dpi=1200,figsize=(10,8)) 
patches, texts,_ =plt.pie(agesRisk.Porcentaje.values,autopct=fmt,pctdistance=1.1)
plt.legend(patches, labels,loc='center left', title='Edad')
plt.axis('equal')
plt.title('Grupo de riesgo\n')
plt.tight_layout()
plt.show()
plt.savefig(config.ROOTPATH+'congresses/figures/canarias_distr_edad_pie_risk.png',dpi=1200)

labels = [f'{group}' for group in agesGP['Grupo edad']]
plt.figure(dpi=1200,figsize=(10,8)) 
patches, texts, _ =plt.pie(agesGP.Porcentaje.values,autopct=fmt,pctdistance=1.1)
plt.legend(patches, labels,loc='center left', title='Edad')
plt.axis('equal')
plt.title('Población General\n')
plt.tight_layout()
plt.show()
plt.savefig(config.ROOTPATH+'congresses/figures/canarias_distr_edad_pie_general.png',dpi=1200)
X.drop('AGE_85+',axis=1,inplace=True)
#%%
def explain_example(lower,upper, preds, coefs, X, descriptions, random_state):
    patient_A=preds.loc[preds.PRED>lower].loc[preds.PRED<upper].sample(1,random_state=random_state)
    patient_A_CCS=X.loc[X.PATIENT_ID==patient_A.PATIENT_ID.values[0]]
    patient_A_CCS=[c for c in patient_A_CCS if patient_A_CCS[c].values[0]==1]
    smallest=coefs[patient_A_CCS].T.nsmallest(5,0)
    df_A=pd.concat([coefs[patient_A_CCS].T.nlargest(5,0),smallest.loc[smallest[0]<0]])
    df_A['CATEGORIES']=df_A.index
    df_A=pd.merge(df_A,descriptions,on='CATEGORIES',how='left')
    agesex_A=X.loc[X.PATIENT_ID==patient_A.PATIENT_ID.values[0]].filter(regex='AGE|FEMALE')
    return patient_A, df_A, agesex_A.T

patient_A, df_A, agesex_A=explain_example(0.95,0.96,preds.loc[preds.TopK==1],coefs,X,descriptions,1)

patient_B, df_B, agesex_B=explain_example(0.5,0.55,preds.loc[preds.TopK==1],coefs,X,descriptions,0)
patient_C, df_C, agesex_C=explain_example(preds.loc[preds.TopK==1].PRED.min(),
                                0.5,preds.loc[preds.TopK==1].loc[preds.OBS2019==1],
                                coefs,X,descriptions,1)

#%%
zeros=pd.DataFrame([np.zeros(X.shape[1]-1)],columns=X.drop('PATIENT_ID',axis=1).columns)
healthy_patients={}
healthy_predictions={}
agecols=[c for c in X if 'AGE' in c]
for c in agecols:
    healthy_patients[f'FEMALE=1{c}']=zeros.copy()
    healthy_patients[f'FEMALE=1{c}'][['FEMALE',c]]=1
    
    healthy_patients[f'FEMALE=0{c}']=zeros.copy()
    healthy_patients[f'FEMALE=0{c}'][[c]]=1
    
    healthy_predictions[f'FEMALE=1{c}']=model.predict_proba(healthy_patients[f'FEMALE=1{c}'])[:,1]
    healthy_predictions[f'FEMALE=0{c}']=model.predict_proba(healthy_patients[f'FEMALE=0{c}'])[:,1]

healthy_patients[f'FEMALE=1AGE_85GT']=zeros.copy()
healthy_patients[f'FEMALE=1AGE_85GT']['FEMALE']=1
healthy_predictions['FEMALE=0AGE_85GT']=model.predict_proba(zeros.copy())[:,1]
healthy_predictions['FEMALE=1AGE_85GT']=model.predict_proba(healthy_patients[f'FEMALE=1AGE_85GT'])[:,1]
#%%
def explain_all_examples(preds, coefs, X, descriptions, healthy_predictions):
    all_explanations={}
    for i,patient_id in enumerate(preds.PATIENT_ID):
        print(i, len(preds.PATIENT_ID))
        CCS=X.loc[X.PATIENT_ID==patient_id]
        # healthy_counterpart=CCS.copy()
        CCS=[c for c in CCS if CCS[c].values[0]==1]
        age=[c for c in CCS if 'AGE' in c]
        age='AGE_85GT' if len(age)==0 else age
        age= age[0] if isinstance(age,list) else age
        sex=f'FEMALE={X.loc[X.PATIENT_ID==patient_id].FEMALE.values[0]}'
        smallest=coefs[CCS].T.nsmallest(5,0)
        df=pd.concat([coefs[CCS].T.nlargest(5,0),smallest.loc[smallest[0]<0]])
        df['CATEGORIES']=df.index
        df=pd.merge(df,descriptions,on='CATEGORIES',how='left')
        pred=preds.loc[preds.PATIENT_ID==patient_id].PRED
        obs=preds.loc[preds.PATIENT_ID==patient_id].OBS
        obs2019=preds.loc[preds.PATIENT_ID==patient_id].OBS2019
        # healthy_counterpart[[c for c in healthy_counterpart.columns if (('CCS' in c) or ('PHARMA' in c))]]=0
        baseline=healthy_predictions[f'{sex}{age}'][0]
        df['type']=np.where(df[0]>=0,'Risk: ', 'Protection: ')
        factors=np.where(df.LABELS.isna(),df.type+df.CATEGORIES, df.type+df.LABELS)
        death='2018' if obs.values[0]==1 else 'unknown'
        death='2019' if obs2019.values[0]==1 else death
        patient_descr=f'Sex: {sex}, Age: {age}, Death : {death}. Prediction is {round(pred.values[0],2)}, {round(pred.values[0]/baseline)} times the risk for a healthy patient of same age and sex.'
        patient_descr=list([patient_descr])+list(['; '.join(factors)])
        all_explanations[patient_id]=patient_descr
        
    all_explanations=pd.DataFrame.from_dict(all_explanations, orient='index', columns=['DESCRIPTION','EXPLANATION'])
    all_explanations['PATIENT_ID']=all_explanations.index
    return all_explanations

explanations_filename=config.PREDPATH+'/explanations_top20k_5.csv'
try: 
    all_explanations=pd.read_csv(explanations_filename)
except:
    all_explanations=explain_all_examples(preds.loc[preds.TopK==1], coefs, X, descriptions, healthy_predictions)        
    all_explanations.to_csv(explanations_filename)        


all_explanations.loc[all_explanations.DESCRIPTION.str.contains(r'^(?=.*FEMALE=1)(?=.*4554)(?=.*ovary)')]

middle_aged_women_cancer_ovary=all_explanations.loc[all_explanations.DESCRIPTION.str.contains(r'^(?=.*FEMALE=1)(?=.*4554)')].loc[all_explanations.EXPLANATION.str.contains(r'ovary')]
cystic_fibrosis=all_explanations.loc[all_explanations.DESCRIPTION.str.contains(r'0004|0511|1217|1834|3544|4554|5564|6569|7074|7579')].loc[all_explanations.EXPLANATION.str.contains(r'Cystic')]
print(middle_aged_women_cancer_ovary.drop('PATIENT_ID',axis=1).to_markdown(index=False))

print(cystic_fibrosis.drop('PATIENT_ID',axis=1).to_markdown(index=False))

very_young=all_explanations.loc[all_explanations.DESCRIPTION.str.contains(r'0004|0511|1217|1834|3544')].loc[all_explanations.DESCRIPTION.str.contains('2018|2019')]

neurologic_organ=all_explanations.loc[all_explanations.DESCRIPTION.str.contains('FEMALE=0')].loc[all_explanations.EXPLANATION.str.contains('^(?=.*[cC]erebrovascular)(?=.*Diabetes mellitus)(?=.*astrointestinal)')]
print(neurologic_organ.drop('PATIENT_ID',axis=1).to_markdown(index=False))

"""
EJEMPLO CÁNCER
 Sex: FEMALE=1, Age: AGE_4554, Death : 2019. 
 Prediction is 0.3, 221 times the risk for a healthy patient of same age and sex.  
| Risk: Secondary malignancies; Risk: Cancer of ovary; Risk: CCSONCOLO; 
Risk: Meningitis (except that caused by tuberculosis or sexually transmitted disease);
 Risk: PHARMA_Steroidresponsive_disease; 
 Protection: AGE_4554; Protection: FEMALE; Protection: PHARMA_Hyperlipidaemia
 
 
EJEMPLO PERSONA JOVEN QUE MURIÓ
Sex: FEMALE=1, Age: AGE_3544, Death : 2019. 
Prediction is 0.25, 539 times the risk for a healthy patient of 
same age and sex.  | 
Risk: Substance-related disorders; Risk: HIV infection;
 Risk: Alcohol-related disorders; Risk: PHARMA_Psychotic_illness;
 Risk: Congestive heart failure; nonhypertensive; 
 Protection: AGE_3544; Protection: FEMALE;
 Protection: PHARMA_Ischaemic_heart_disease_hypertension |
 
EJEMPLO SIN CÁNCER (ENFERMO DE ÓRGANO Y NEUROLÓGICO)
Sex: FEMALE=0, Age: AGE_85GT, Death : unknown.
 Prediction is 0.31, 5 times the risk for a healthy 
 patient of same age and sex. |
 Risk: Chronic ulcer of skin; 
 Risk: Acute cerebrovascular disease;
 Risk: Chronic obstructive pulmonary disease and
 bronchiectasis; Risk: PHARMA_Diabetes;
 Risk: Diabetes mellitus with complications; 
 Protection: PHARMA_Hyperlipidaemia; 
 Protection: PHARMA_Hypertension; 
 Protection: PHARMA_Ischaemic_heart_disease_hypertension 

Sex: FEMALE=0, Age: AGE_85GT, Death : unknown. 
Prediction is 0.22, 3 times the risk for a healthy patient of same age and sex.
 | Risk: Acute cerebrovascular disease; Risk: Deficiency and other anemia;
 Risk: PHARMA_Diabetes; Risk: Diabetes mellitus with complications; 
 Risk: Gastrointestinal hemorrhage; 
 Protection: PHARMA_Ischaemic_heart_disease_hypertension 
"""
 