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

#%%
fname1='gma_needs_manual_revision_icd9.csv'
fname2='gma_needs_manual_revision_icd10.csv'
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

def guessingCCS(missingDX, dictionary):
    success, failure={},{}
    for dx in missingDX.CODE:
        print(dx)
        if not isinstance(dx,str):
            continue
        if dx=='ONCOLO':
            continue
        i=0
        options=list(dictionary.loc[dictionary.CODE.str.startswith(dx.upper())].CCS.unique())
        code=dx
        while (len(options)==0) and (i<=len(dx)):
            i+=1
            code=dx[:-i]
            options=list(dictionary.loc[dictionary.CODE.str.startswith(code)].CCS.unique())

        if '259' in options and len(options)>=2: options.remove('259') #CCS 259 is for residual unclassified codes
        if len(options)==1:
            success[(dx, code)]=options[0]
        else:
            failure[(dx, code)]=options
        # break
    print(failure)
    return(success, failure)
def assign_success(success, dic):
    if len(success.keys())>0:
        assign_success={'CODE':[k[0] for k in success.keys()], 'CCS':[v for v in success.values()]}
        assign_success['CODE'].append('ONCOLO')
        assign_success['CCS'].append('ONCOLO')
        dic= pd.concat([dic, pd.DataFrame(assign_success)],ignore_index=True)
    return dic
def suggestCCS(revisiondf, success):   
    success_v2={k[0]:v for k,v in success.items()}
    revisiondf['CCS_suggestion']=np.nan
    for code in revisiondf.CODE:
        if code in success_v2.keys():
            revisiondf.loc[revisiondf.CODE==code,'CCS_suggestion']=success_v2[code]
    return revisiondf
            
#%%
icd10cm=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                    dtype=str,)
icd9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD9']), dtype=str,
                 usecols=['ICD-9-CM CODE','CCS LVL 1 LABEL','CCS LVL 2 LABEL',
                          'CCS LVL 3 LABEL','CCS LVL 4 LABEL'])
icd10cm=text_preprocessing(icd10cm,
                           [c for c in icd10cm if ((not 'DESCRIPTION' in c) and (not 'LABEL' in c))],
                           mode='icd10cm')
icd9=text_preprocessing(icd9, [c for c in icd9 if (not 'LABEL' in c)],mode='icd9')
#%%
if (not Path(fname1).is_file()) or (not Path(fname2).is_file()):
    """ READ EVERYTHING """ 
    
    diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2016]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                      index_col=False)
    diags2=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2017]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                      index_col=False)
    #%%
    """ TEXT PREPROCESSING """
    
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
    assert all(diags.isnull().sum()==0), 'Null values encountered after cleaning up config.ICDFILES'
    
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
    f10=os.path.join(config.INDISPENSABLEDATAPATH,f'ccs/manually_revised_icd10.csv')
    f9=os.path.join(config.INDISPENSABLEDATAPATH,f'ccs/manually_revised_icd9.csv')
    manual_revision_ccs_9=pd.read_csv(f9, usecols=['CODE','NEW_CODE'])
    manual_revision_ccs_10=pd.read_csv(f10, usecols=['CODE','NEW_CODE'])


    revision_9=pd.merge(revision_9,manual_revision_ccs_9,on='CODE', how='left').drop_duplicates('CODE')
    revision_10=pd.merge(revision_10,manual_revision_ccs_10,on='CODE', how='left').drop_duplicates('CODE')

    print(f'ICD 9: {revision_9.NEW_CODE.isna().sum()} codes to revise')
    print(f'ICD 10: {revision_10.NEW_CODE.isna().sum()} codes to revise')

    #%%
    revision_9[['N','CODE','NEW_CODE']].to_csv(fname1, index=False)
    revision_10[['N','CODE','NEW_CODE']].to_csv(fname2, index=False)
else:
    revision_9=pd.read_csv(fname1)
    revision_10=pd.read_csv(fname2)
#%%
rev9fname='revision_9_with_ccs_suggestions.csv'
rev10fname='revision_10_with_ccs_suggestions.csv'
#%%
if not Path(rev9fname).is_file():
    success9, failure9=guessingCCS(revision_9,icd9)
    icd9=assign_success(success9,icd9)
    revision_9_bis=suggestCCS(revision_9, success9)
    df=revision_9_bis.loc[(revision_9_bis.NEW_CODE.isna())].loc[(revision_9_bis.CCS_suggestion.isna())]
    print(f'Lacking CCS suggestions for {len(df)} codes')
    revision_9_bis.to_csv(rev9fname, index=False)
else:
    revision_9_bis=pd.read_csv(rev9fname)
#%%
if not Path(rev10fname).is_file():
    success10, failure10=guessingCCS(revision_10,icd10cm)
    icd10=assign_success(success10,icd10cm)
    revision_10_bis=suggestCCS(revision_10, success10)
    df=revision_10_bis.loc[(revision_10_bis.NEW_CODE.isna())].loc[(revision_10_bis.CCS_suggestion.isna())]
    print(f'Lacking CCS suggestions for {len(df)} codes')
    revision_10_bis.to_csv('revision_10_with_ccs_suggestions.csv', index=False)
else:
    revision_10_bis=pd.read_csv(rev10fname)
    

#%%
last_revision_9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,'ccs','diccionario_cie9_previo_GMA_V3.csv'))
last_revision_9['CODE']=last_revision_9.cie9
revision_9_bis=pd.merge(revision_9_bis, last_revision_9, on='CODE', how='left')
revision_9_bis['CODE_Edu']=np.where(revision_9_bis.newcie9.isna(),revision_9_bis.CODE,revision_9_bis.newcie9)
revision_9_bis['CODE_original']=revision_9_bis.CODE
revision_9_bis=revision_9_bis[['N', 'CODE_original','CODE_Edu','CCS_suggestion']]
revision_9_bis=revision_9_bis.rename(columns={'CODE_Edu':'CODE'})
missing9=missingDX(icd9, revision_9_bis)
revision_9_bis_withccs=pd.merge(revision_9_bis,icd9[['CODE','CCS']],on='CODE', how='left')
lo_que_cuelga=revision_9_bis_withccs.loc[~revision_9_bis_withccs.CCS_suggestion.isna()]
no_coincide=lo_que_cuelga.loc[lo_que_cuelga.CCS_suggestion.astype(float)!=lo_que_cuelga.CCS.astype(float)].dropna(subset=['CCS','CCS_suggestion'])
no_coincide.to_csv('no_coincide_ccs_cie9.csv', index=False)
#%%
last_revision_10=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,'ccs','diccionario_cie10_previo_GMA_V2.csv'))
last_revision_10['CODE']=last_revision_10.cie10
revision_10_bis=pd.merge(revision_10_bis, last_revision_10, on='CODE', how='left')
#%%
revision_10_bis['CODE_Edu']=np.where(revision_10_bis.newcie10.isna(),revision_10_bis.CODE,revision_10_bis.newcie10)
revision_10_bis['CODE_original']=revision_10_bis.CODE
#%%
revision_10_bis=revision_10_bis[['N', 'CODE_original','CODE_Edu','CCS_suggestion']]
revision_10_bis=revision_10_bis.rename(columns={'CODE_Edu':'CODE'})
missing10=missingDX(icd10cm, revision_10_bis)
#%%
revision_10_bis_withccs=pd.merge(revision_10_bis,icd10cm[['CODE','CCS']],on='CODE', how='left')
#%%
lo_que_cuelga=revision_10_bis_withccs.loc[~revision_10_bis_withccs.CCS_suggestion.isna()]
no_coincide=lo_que_cuelga.loc[lo_que_cuelga.CCS_suggestion.astype(float)!=lo_que_cuelga.CCS.astype(float)].dropna(subset=['CCS','CCS_suggestion'])
no_coincide.to_csv('no_coincide_ccs_cie10.csv', index=False)

#%%
"""
CHANGE THE BIG DIAGNOSIS DATAFRAME, 
MODIFYING CODES PRESENT IN CODE_original WITH THE
CORRESPONDING MANUALLY REVISED NEW CODE
"""

# step 1: read and preprocess diagnoses
diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2016]),
                  usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                  index_col=False)
diags2=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[2017]),
                  usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                  index_col=False)
#%%
""" TEXT PREPROCESSING """

diags=text_preprocessing(diags, ['CIE_VERSION','CIE_CODE'],mode='diags')
diags2=text_preprocessing(diags2, ['CIE_VERSION','CIE_CODE'],mode='diags')
#%%
correct_diags_2016,correct_diags_2017=pd.DataFrame(),pd.DataFrame()
fullrevision=pd.concat([revision_9_bis,revision_10_bis])
fullrevision.loc[fullrevision.CODE=='no pongo nada','CODE']=np.nan
fullrevision.loc[fullrevision.CODE=='NO HAGO NADA','CODE']=np.nan
#%%
""" USE THE MANUAL REVISION TO CHANGE DIAGNOSTIC CODES WHEN NECESSARY """
correct_diags_2016_fname=os.path.join(config.INDISPENSABLEDATAPATH,'ccs/gma_dx_in_2016.txt')
if not Path(correct_diags_2016_fname).is_file():
    correct_diags_2016=diags.copy()
    #Those with no NEW_CODE specified are lost -> discard rows with NAs
    L0=len(correct_diags_2016)
    for code, new in zip(fullrevision.CODE_original, fullrevision.CODE):
        correct_diags_2016.loc[correct_diags_2016.CODE==code, 'CODE']=new
    correct_diags_2016=correct_diags_2016.dropna(subset=['CODE'])
    L=len(correct_diags_2016)
    print(f'We have lost {L0-L} diagnoses that still have no CCS')
    correct_diags_2016.to_csv(correct_diags_2016_fname,index=False)
else:
    correct_diags_2016=pd.read_csv(correct_diags_2016_fname)
#%%
correct_diags_2017_fname=os.path.join(config.INDISPENSABLEDATAPATH,'ccs/gma_dx_in_2017.txt')
if not Path(correct_diags_2016_fname).is_file():
    correct_diags_2017=diags2.copy()
    #Those with no NEW_CODE specified are lost -> discard rows with NAs
    L0=len(correct_diags_2017)
    for code, new in zip(fullrevision.CODE_original, fullrevision.CODE):
        correct_diags_2017.loc[correct_diags_2017.CODE==code, 'CODE']=new
    correct_diags_2017=correct_diags_2017.dropna(subset=['CODE'])
    L=len(correct_diags_2017)
    print(f'We have lost {L0-L} diagnoses that still have no CCS')
    correct_diags_2017.to_csv(correct_diags_2017_fname,index=False)
else:
    correct_diags_2017=pd.read_csv(correct_diags_2017_fname)
assert False
#%%
""" CHECK CCS ASSIGNMENT """
""" ASSIGN CCS CATEGORIES TO DIAGNOSTIC CODES """
dict9=icd9[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
dict10=icd10cm[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
dict10=pd.concat([dict10, 
                  pd.DataFrame.from_dict({'CODE':['ONCOLO'], 'CCS':['ONCOLO'], 
                                          'DESCRIPTION': ['Undetermined oncology code'],
                                          'CIE_VERSION':['10']})])
fulldict=pd.concat([dict9,dict10]).drop_duplicates()
#Drop codes that were not diagnosed to any patient in the current year
diags_with_ccs_2016=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'CCS':[], 'DESCRIPTION':[]})
# df=diags.copy()
diags_with_ccs_2016=pd.merge(correct_diags_2016, fulldict, on=['CODE','CIE_VERSION'], how='inner')#[['PATIENT_ID','CODE','CCS']]
