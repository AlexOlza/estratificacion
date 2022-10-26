#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re

DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
INDISPENSABLEDATAPATH=DATAPATH+'indispensable/'
yr=2016

# df=pd.read_csv(os.path.join(INDISPENSABLEDATAPATH,'ccs','diccionario_ATC_farmacia.csv'))

def text_preprocessing(df, col):
    df[col]=df[col].str.upper()
    df[col]=df[col].str.replace(r'[^a-zA-Z\d]', r'',regex=True).values #drop non-alphanumeric
    return df
def generateCCSData(yr,  X,
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
        cols_to_merge=['PATIENT_ID']+[c for c in X if (('AGE' in c) or ('FEMALE' in c)) or ('CCS' in c)]
        Xx=pd.merge(X, Xatc, on=cols_to_merge, how='outer')
        return Xx
    #%%
    """ FUNCTIONS """
 
    #%%
    """ READ EVERYTHING """ 
    atc_dict=pd.read_csv(os.path.join(INDISPENSABLEDATAPATH,'ccs',
                                     'diccionario_ATC_farmacia.csv'))
    rx=pd.read_csv(os.path.join(INDISPENSABLEDATAPATH,'ccs',f'rx_in_{yr}.txt'), 
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
    for start in atc_dict.starts_with.drop_duplicates(): 
        # print(atc_dict.loc[atc_dict.starts_with==start, 'drug_group'].values[0])
        unique_codes_prescribed.loc[unique_codes_prescribed.CODE.str.startswith(start),'drug_group']=atc_dict.loc[atc_dict.starts_with==start, 'drug_group'].values[0]
    n_distinct_drugs=len([c for c in unique_codes_prescribed.drug_group.unique() if c!=''])
    print(f'In year {yr}, {n_distinct_drugs} distinct drug groups from the dictionary were prescribed to patients')
    print(f'We have not found a group for {len(unique_codes_prescribed)-n_distinct_drugs} distinct codes')
    print(f'{len(atc_dict.starts_with)-n_distinct_drugs} dictionary entries have not been used')
    
    #Drop codes that were not prescribed to any patient in the current year
    rx_with_drug_group=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'drug_group':[]})
    # df=diags.copy()
    rx_with_drug_group=pd.merge(rx, unique_codes_prescribed, on=['CODE'], how='inner')[['PATIENT_ID','CODE','drug_group']]
    withoutna=rx_with_drug_group.dropna()
    print()33793525-
    #%%
    """ ASSIGN CCS CATEGORIES TO DIAGNOSTIC CODES """
    dict9=icd9[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
    dict10=icd10cm[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
    dict10=pd.concat([dict10, 
                      pd.DataFrame.from_dict({'CODE':['ONCOLO'], 'CCS':['ONCOLO'], 
                                              'DESCRIPTION': ['Undetermined oncology code'],
                                              'CIE_VERSION':['10']})])
    fulldict=pd.concat([dict9,dict10]).drop_duplicates()
    #Drop codes that were not diagnosed to any patient in the current year
    diags_with_ccs=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'CCS':[], 'DESCRIPTION':[]})
    # df=diags.copy()
    diags_with_ccs=pd.merge(diags, fulldict, on=['CODE','CIE_VERSION'], how='inner')[['PATIENT_ID','CODE','CCS']]
    
    #%%
    """ COMPUTE THE DATA MATRIX """
    i=0
    import time
    t0=time.time()
    try:
        X.set_index('PATIENT_ID', inplace=True)
    except KeyError:
        pass
    for ccs_number, df in diags_with_ccs.groupby('CCS'):
        # print(ccs_number,df['DESCRIPTION'].values[0])
        amount_per_patient=df.groupby('PATIENT_ID').size().to_frame(name=f'CCS{ccs_number}')
        X[f'CCS{ccs_number}']=np.int16(0)
        
        X.update(amount_per_patient)
        X[f'CCS{ccs_number}'].fillna(0,axis=0,inplace=True)
        print(f'CCS{ccs_number}', X[f'CCS{ccs_number}'].sum())
        i+=1
    X.reset_index()
    print('TIME : ' , time.time()-t0)
 
    
    print(f'{i} dfs processed')
    
    X.reindex(sorted(X.columns), axis=1).to_csv(filename)
    print('Saved ',filename)
    return 0,0
