#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:20:27 2021

@author: aolza
"""
import os
import pandas as pd
import time
import numpy as np
from pathlib import Path
from python_settings import settings as config
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng,assertMissingCols, report_transform
from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load, create_fullacgfiles

def listIntersection(a,b): return list(set(a) & set(b))

def generateCCSData(yr,  X,
            **kwargs):
   
    predictors=kwargs.get('predictors',None)
    PHARMACY=kwargs.get('PHARMACY', False)
    
    filename=os.path.join(config.DATAPATH,config.CCSFILES[yr])
    #%%
    """ FUNCTIONS """
    def text_preprocessing(df, columns, mode):
        allowed=['icd9','icd10cm','diags']
        assert mode in allowed, f'Allowed mode values are: {allowed}'
        for c in columns:
            df[c]=df[c].astype(str)
            df[c]=df[c].str.replace(r'[^a-zA-Z\d]', r'',regex=True).values #drop non-alphanumeric
            df[c]=df[c].str.upper()
        if mode=='icd9':
            #the CCS category is expressed between brackets and followed by a dot,
            #and can be inside any column of type 'CCS LVL x LABEL'.
            df.rename(columns={'ICD-9-CM CODE':'CODE'},inplace=True)
            df['CCS']=df['CCS LVL 1 LABEL']+df['CCS LVL 2 LABEL']+df['CCS LVL 3 LABEL']+df['CCS LVL 4 LABEL']
            df.CCS=df.CCS.str.extract(r'(?P<CCS>[0-9]+)').CCS
            df.CODE=df.CODE.str.slice(0,5)
            df['CIE_VERSION']='9'
            df['DESCRIPTION']=df['CCS LVL 1 LABEL']+df['CCS LVL 2 LABEL']+df['CCS LVL 3 LABEL']+df['CCS LVL 4 LABEL']
            
        elif mode=='icd10cm':
            df.rename(columns={'ICD-10-CM CODE':'CODE', 'CCS CATEGORY':'CCS'},inplace=True)
            df.CODE=df.CODE.str.slice(0,6)
            df['CIE_VERSION']='10'
            df.rename(columns={'CCS CATEGORY DESCRIPTION':'DESCRIPTION'},inplace=True)
            
        else:
            #In the diagnoses dataset, ICD10CM dx that start with a digit are related to oncology
            df.loc[(df.CIE_VERSION.astype(str).str.startswith('10') & df.CIE_CODE.str.match('^[0-9]')),'CIE_CODE']='ONCOLOGY' 
            #ICD10CM and ICD9 only allow for 6 and 5 characters respectively
            df.loc[df.CIE_VERSION.str.startswith('9'),'CIE_VERSION']='9'
            df.loc[df.CIE_VERSION.str.startswith('10'),'CIE_VERSION']='10'
            df.loc[df.CIE_VERSION=='10','CIE_CODE']=df.loc[df.CIE_VERSION=='10','CIE_CODE'].str.slice(0,6)
            df.loc[df.CIE_VERSION=='9','CIE_CODE']=df.loc[df.CIE_VERSION=='9','CIE_CODE'].str.slice(0,5)
            print('Dropping NULL codes:')
            print(df.loc[df.CIE_CODE.isnull()])
            df.dropna(subset=['CIE_CODE'], inplace=True)
            df.rename(columns={'CIE_CODE':'CODE'},inplace=True)
        return df 
           
    def missingDX(dic,diags):
        diagsCodes=set(diags.CODE)
        dictCodes=set(dic.CODE)
        return(diagsCodes-dictCodes)
    def needsManualRevision(failure, dictionary, appendix='',
                         diags=None,
                         exclude={}):      
        fname=f'gma_needs_manual_revision{appendix}.csv'
        if Path(fname).is_file():
            print('Reading ',fname)
            already_there=pd.read_csv(fname)
        else:
            already_there=pd.DataFrame({'CODE':[],'N':[]})
        keys_to_revise=list(failure)
        
        
        
        if not all([k in already_there.CODE.values for k in keys_to_revise]):
            
            new_codes={}
            for i,key in enumerate(keys_to_revise):
                print(i)
                if not key in already_there.CODE.values: 
                    N=len(diags.loc[diags.CODE==key].PATIENT_ID.unique())
                    new_codes[i]= [key,N ]    
            already_there=already_there.append(pd.DataFrame.from_dict(new_codes,orient='index',columns=['CODE','N']))
            already_there.N=already_there.N.astype(int)
            already_there=already_there.sort_values('N',ascending=False)
            already_there.to_csv(fname, index=False)
        return(already_there)
    #%%
    """ READ EVERYTHING """ 
    icd10cm=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str,)
    icd9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD9']), dtype=str,
                     usecols=['ICD-9-CM CODE','CCS LVL 1 LABEL','CCS LVL 2 LABEL',
                              'CCS LVL 3 LABEL','CCS LVL 4 LABEL'])
    diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2016]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                      index_col=False)
    diags2=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2017]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                      index_col=False)
    #%%
    """ TEXT PREPROCESSING """
    icd10cm=text_preprocessing(icd10cm,
                               [c for c in icd10cm if ((not 'DESCRIPTION' in c) and (not 'LABEL' in c))],
                               mode='icd10cm')
    icd9=text_preprocessing(icd9, [c for c in icd9 if (not 'LABEL' in c)],mode='icd9')
    diags=text_preprocessing(diags, ['CIE_VERSION','CIE_CODE'],mode='diags')
    diags2=text_preprocessing(diags2, ['CIE_VERSION','CIE_CODE'],mode='diags')
    #%%
    diags=pd.concat([diags, diags2])
    #%%
    """ ICD 10 CM ASSERTIONS"""
    #Check no null values at reading time
    assert all(icd10cm.isnull().sum()==0), f'Null values encountered when reading {config.ICDTOCCSFILES["ICD10CM"]}'
    assert icd10cm.CCS.isnull().sum()==0, 'Some codes in the ICD10CM dictionary have not been assigned a CCS :('
    
    """ ICD 9 ASSERTIONS"""
    #Check no null values at reading time
    assert all(icd9.isnull().sum()==0), f'Null values encountered when reading {config.ICDTOCCSFILES["ICD9"]}' 
    assert icd9.CCS.describe().isnull().sum()==0, 'Some codes in the ICD9 dictionary have not been assigned a CCS :('

    """ DIAGNOSES ASSERTIONS""" 
    #Check null values
    assert all(diags.isnull().sum()==0), f'Null values encountered after cleaning up {config.ICDFILES[yr]}'

    #%%
    """ PERFORM MANUAL REVISION ON MISSING CODES """
    missing_in_icd9=missingDX(icd9,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('9')])
    missing_in_icd10cm=missingDX(icd10cm,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10')])
    print('Missing quantity ICD9: ', len(missing_in_icd9))
    print('Missing quantity ICD10: ', len(missing_in_icd10cm))
    
    #%%
    revision_9=needsManualRevision(missing_in_icd9, icd9, appendix='_icd9', diags=diags)
    #%%
    revision_10=needsManualRevision(missing_in_icd10cm, icd10cm, appendix='_icd10', diags=diags)
#%%
if __name__=='__main__':
    import sys
    sys.path.append('/home/aolza/Desktop/estratificacion/')
    yr=2016
