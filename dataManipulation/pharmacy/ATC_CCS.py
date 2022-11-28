#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re

from pathlib import Path
from python_settings import settings as config

import configurations.utility as util
configuration=util.configure()
from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load, create_fullacgfiles

def text_preprocessing(df, col):
    df[col]=df[col].str.upper()
    df[col]=df[col].str.replace(r'[^a-zA-Z\d]', r'',regex=True).values #drop non-alphanumeric
    return df

def ATC_drug_group_descriptive(rx_with_drug_group, yr):
    number_of_cases={}
    for code, df in rx_with_drug_group.groupby('CODE'):
        print(code)
        number_of_cases[code]=[len(df.PATIENT_ID.unique())]
    number_of_cases=pd.DataFrame.from_dict(number_of_cases,orient='index',columns=['N'])    
    number_of_cases['CODE']=number_of_cases.index
    unique_rx_with_group=rx_with_drug_group.drop_duplicates('CODE')[['CODE','drug_group']]
    df=pd.merge(unique_rx_with_group,number_of_cases,on='CODE').sort_values('N', ascending=False)
    df.to_excel(f'{config.ROOTPATH}dataManipulation/pharmacy/pharmacy_descriptive_{yr}.xlsx')

def generatePharmacyData(yr,  X, binarize=False,
            **kwargs):
    
    """ CHECK IF THE MATRIX IS ALREADY ON DISK """
    predictors=kwargs.get('predictors',None)
    filename=os.path.join(config.DATAPATH,config.ATCFILES[yr])
    
    if Path(filename).is_file():
        print('X number of columns is  ',len(X.columns))
        Xatc=load(config.ATCFILES[yr],directory=config.DATAPATH,
                    predictors=predictors)
        print('Xatc number of columns is ',len(Xatc.columns) )
        assert 'PATIENT_ID' in X.columns
        assert 'PATIENT_ID' in Xatc.columns
        cols_to_merge=['PATIENT_ID']
        if binarize:
            predictors=[c for c in Xatc if not c=='PATIENT_ID']
            print('Binarizing pharmacy...')
            Xatc[predictors]=(Xatc[predictors]>0).astype(int)
        Xx=pd.merge(X, Xatc, on=cols_to_merge, how='inner')
        return Xx

    #%%
    """ READ EVERYTHING """ 
    atc_dict=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,'ccs',
                                     'diccionario_ATC_farmacia.csv'))
    rx=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,'ccs',f'rx_in_{yr}.txt'), 
                   names=['PATIENT_ID','date','CODE','a','number' ])
    #%%
    """ TEXT PREPROCESSING """
    rx=text_preprocessing(rx, 'CODE')
    atc_dict=text_preprocessing(atc_dict, 'starts_with')
    #%%
    """ ASSIGN DRUG GROUP TO CODES """
    #Unique codes prescribed in the current year
    unique_codes_prescribed=pd.DataFrame({'CODE':rx.CODE.drop_duplicates()})
    import numpy as np
    unique_codes_prescribed['drug_group']=np.nan
    for start in atc_dict.starts_with.drop_duplicates().sort_values(key=lambda x: x.str.len()): 
        unique_codes_prescribed.loc[unique_codes_prescribed.CODE.str.startswith(start),'drug_group']=atc_dict.loc[atc_dict.starts_with==start, 'drug_group'].values[0]
    n_distinct_drugs=len([c for c in unique_codes_prescribed.drug_group.dropna().unique()])
  
    print(f'In year {yr}, {n_distinct_drugs} distinct drug groups from the dictionary were prescribed to patients')
    print(f'We have not found a group for {len(unique_codes_prescribed)-n_distinct_drugs} distinct codes')
    print(f'{len(atc_dict.drug_group.unique())-n_distinct_drugs} dictionary entries have not been used')
    
    #Drop codes that were not prescribed to any patient in the current year
    rx_with_drug_group=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'drug_group':[]})
    # df=diags.copy()
    rx_with_drug_group=pd.merge(rx, unique_codes_prescribed, on=['CODE'], how='inner')[['PATIENT_ID','CODE','drug_group']].dropna()
    ATC_drug_group_descriptive(rx_with_drug_group,yr)
    # del rx

    #%%
    """ COMPUTE THE DATA MATRIX """
    i=0
    import time
    t0=time.time()
    try:
        X.set_index('PATIENT_ID', inplace=True)
    except KeyError:
        pass
    for group_, df in rx_with_drug_group.groupby('drug_group'):
        group=re.sub("[^a-zA-Z\d_]",'',re.sub('\s', '_', group_))
        amount_per_patient=df.groupby('PATIENT_ID').size().to_frame(name=f'PHARMA_{group}')
        X[f'PHARMA_{group}']=np.int16(0)
        
        X.update(amount_per_patient)
        X[f'PHARMA_{group}'].fillna(0,axis=0,inplace=True)
        print(f'{group}', X[f'PHARMA_{group}'].sum())
        i+=1
    X.reset_index(inplace=True)
    print('TIME : ' , time.time()-t0)
 
    
    print(f'{i} dfs processed')
    
    print('Condition 1: Congestive_heart_failure must have at least one in every block')
    X_heart_fail=X.filter(regex='PATIENT_ID|PHARMA_Congestive_heart_failure', axis=1)
    condition1=X_heart_fail.drop(['PATIENT_ID'],axis=1).min(axis=1)>0
    for block in [1,2,3]:
        col=f'PHARMA_Congestive_heart_failure_block_{block}'
        X.loc[~condition1,col]=0
        
    print('Condition 2: Benign_prostatic_hyperplasia must be zero for females')
    X.loc[X.FEMALE==1, 'PHARMA_Benign_prostatic_hyperplasia']=0
    # sum_heart_fail_meds=X_heart_fail.drop(['PATIENT_ID'],axis=1).sum(axis=1)
    # X_heart_fail['Congestive_heart_failure']=np.where(condition1,sum_heart_fail_meds,0)
    
   
    X.reindex(sorted(X.columns), axis=1).to_csv(filename, index=False)
    print('Saved ',filename)
    X_binary=X.drop(['FEMALE','PATIENT_ID'],axis=1).copy()
    X_binary=(X_binary>0).astype(int)
    n_patients=(X_binary.sum(axis=1)>=1).sum()
    print('Number of patients with at least one prescription (%): ',n_patients, n_patients*100/len(X))
    
    n_patients_per_group=X_binary.sum()
    n_patients_per_group.to_excel(f'{config.ROOTPATH}dataManipulation/pharmacy/patients_per_group_{yr}.xlsx')
    return X
