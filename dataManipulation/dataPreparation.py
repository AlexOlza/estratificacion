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

import configurations.utility as util
configuration=util.configure()
from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng,assertMissingCols, report_transform
from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load, create_fullacgfiles
# from matplotlib_venn import venn2
def listIntersection(a,b): return list(set(a) & set(b))
# FIXME not robust!!!
def excludeHosp(df,filtros,criterio):
    filtros=['{0}_{1}'.format(f,criterio) for f in filtros]

    assertMissingCols(filtros+[criterio],df,'exclude')
    
    anyfiltros=(df[filtros].sum(axis=1)>=1)
    crit=(df[criterio]>=1)
    
    df['remove']=np.where((anyfiltros & crit),1,0)

    df[criterio]=df[criterio]-df['remove']

    
# TODO MOVE ASSERTIONS BEFORE LOADING BIG FILES!!!!
#OLDBASE IS OBSOLETE
def getData(yr,columns=config.COLUMNS,
            predictors=config.PREDICTORREGEX,
            exclude=config.EXCLUDE,
            resourceUsage=config.RESOURCEUSAGE,
            **kwargs):
    try:
        fullEDCs = kwargs.get('fullEDCs', config.FULLEDCS)
    except AttributeError:
        fullEDCs = False
        
    try:
        CCS = kwargs.get('CCS', config.CCS)
    except AttributeError:
        CCS = False
        
    oldbase = kwargs.get('oldbase', False)
    
    if oldbase or ('COSTE_TOTAL_ANO2' in columns):
        if 'ingresoUrg' not in predictors and not ('COSTE_TOTAL_ANO2' in columns):
            predictors=predictors+'|ingresoUrg'
            response='ingresoUrg'
        elif ('COSTE_TOTAL_ANO2' in columns):
            predictors=predictors+'|COSTE_TOTAL_ANO2'
            response='COSTE_TOTAL_ANO2'
        if fullEDCs:
            acg=create_fullacgfiles(config.FULLACGFILES[yr],yr,directory=config.DATAPATH,
                    predictors=predictors)
            coste=load(filename=config.ACGFILES[yr],predictors=r'PATIENT_ID|COSTE_TOTAL_ANO2')
            acg=pd.merge(acg,coste,on='PATIENT_ID')
        else:
            acg=load(filename=config.ACGFILES[yr],predictors=predictors)
        print('not opening allhospfile')
        X=acg.drop(response,axis=1)
        return(X.reindex(sorted(X.columns), axis=1),acg[['PATIENT_ID',response]])#Prevents bug #1

    cols=columns.copy() 
    t0=time.time()
    ing=loadIng(config.ALLHOSPITFILE,config.DATAPATH)
    ingT=createYearlyDataFrames(ing)
    missing_columns=set([yr,yr+1])-set(ingT.keys())
    assert len(missing_columns)==0,'getData: HOSPITALIZATION DATA FOR YEAR {0} NOT AVAILABLE.'.format(missing_columns)
    del ing
    if exclude:
        assert isinstance(exclude,(str,list)), 'getData accepts str or list or None/False as exclude!'
        exclude=[exclude] if isinstance(exclude,str) else exclude
        for c in columns:
            excludeHosp(ingT[yr+1], exclude, c)
            assert min(ingT[yr+1][c])>=0, 'negative hospitalizations'
        util.vprint('excluded ',exclude)
    report_transform(ingT[yr+1])

    if cols:
        cols=[cols] if isinstance(cols,str) else cols
        if 'PATIENT_ID' not in cols:
            cols.insert(0,'PATIENT_ID')
        assertMissingCols(cols,ingT[yr+1],'getData')
        ingT[yr+1]=ingT[yr+1].loc[:,cols]
    if resourceUsage:
        df16=resourceUsageDataFrames(yr)[yr]
        full16=prepare(df16,indicePrivacion=config.INDICEPRIVACION,yr=yr,verbose=config.VERBOSE,predictors=predictors)
        del df16
    elif fullEDCs:
        full16=create_fullacgfiles(config.FULLACGFILES[yr],yr,directory=config.DATAPATH,
                    predictors=predictors)
    elif CCS:
        Xprovisional=load(filename=config.ACGFILES[yr],predictors=predictors)
        full16=generateCCSData(yr,  Xprovisional)
    else:
        full16=load(filename=config.ACGFILES[yr],predictors=predictors)
   
    assert 'PATIENT_ID' in full16.columns
    pred16=pd.merge(full16,ingT[yr]['PATIENT_ID'],on='PATIENT_ID',how='left')

            
    y17=pd.merge(ingT[yr+1],full16['PATIENT_ID'],on='PATIENT_ID',how='outer').fillna(0)
    print('number of patients y17: ', sum(np.where(y17[config.COLUMNS]>=1,1,0)))
    data=pd.merge(pred16,y17,on='PATIENT_ID')
    print('number of patients in data: ', sum(np.where(data[config.COLUMNS]>=1,1,0)))
    print('getData time: ',time.time()-t0)
    finalcols=listIntersection(data.columns,pred16.columns)
    X,y=data[finalcols].reindex(sorted(data[finalcols].columns), axis=1),data[cols]
    
    return(X[finalcols].reindex(sorted(data[finalcols].columns), axis=1),y[cols])

def generateCCSData(yr,  X,
            **kwargs):
    """ CHECK IF THE MATRIX IS ALREADY ON DISK """
    predictors=kwargs.get('predictors',None)
    filename=os.path.join(config.DATAPATH,config.CCSFILES[yr])
    if Path(filename).is_file():
        print('X number of columns is  ',len(X.columns))
        Xccs=load(config.CCSFILES[yr],directory=config.DATAPATH,
                    predictors=predictors)
        print('Xccs number of columns is ',len(Xccs.columns) )
        assert 'PATIENT_ID' in X.columns
        assert 'PATIENT_ID' in Xccs.columns
        cols_to_merge=['PATIENT_ID']+[c for c in X if (('AGE' in c) or ('FEMALE' in c))]
        Xx=pd.merge(X, Xccs, on=cols_to_merge, how='outer')
        return Xx
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
        elif mode=='icd10cm':
            df.rename(columns={'ICD-10-CM CODE':'CODE', 'CCS CATEGORY':'CCS'},inplace=True)
            df.CODE=df.CODE.str.slice(0,6)
        else:
            #In the diagnoses dataset, ICD10CM dx that start with a digit are related to oncology
            df.loc[(df.CIE_VERSION.astype(str).str.startswith('10') & df.CIE_CODE.str.match('^[0-9]')),'CIE_CODE']='ONCOLOGY' 
            #ICD10CM and ICD9 only allow for 6 and 5 characters respectively
            df.loc[df.CIE_VERSION.str.startswith('9'),'CIE_VERSION']='9'
            df.loc[df.CIE_VERSION.str.startswith('10'),'CIE_VERSION']='10'
            df.loc[df.CIE_VERSION.str.startswith('10'),'CIE_CODE']=df.loc[df.CIE_VERSION.astype(str).str.startswith('10'),'CIE_CODE'].str.slice(0,6)
            df.loc[df.CIE_VERSION.str.startswith('9'),'CIE_CODE']=df.loc[df.CIE_VERSION.astype(str).str.startswith('9'),'CIE_CODE'].str.slice(0,5)
            print('Dropping NULL codes:')
            print(df.loc[df.CIE_CODE.isnull()])
            df.dropna(subset=['CIE_CODE'], inplace=True)
        return df 
           
    def missingDX(dic,diags):
        diagsCodes=set(diags.CIE_CODE)
        dictCodes=set(dic.CODE)
        return(diagsCodes-dictCodes)
    def guessingCCS(missingDX, dictionary):
        success, failure={},{}
        for dx in missingDX:
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
    def needsManualRevision(failure, dictionary, appendix='',
                            diags=None,
                            exclude={}):      
        import csv
        if Path(f'needs_manual_revision{appendix}.csv').is_file():
            already_there=pd.read_csv(f'needs_manual_revision{appendix}.csv',dtype=str)
            mode='a'
        else:
            mode='w'   
            already_there=pd.DataFrame({'CODE':['']})
        keys_to_revise=[k[0] for k in failure.keys()]
        with open(f'needs_manual_revision{appendix}.csv', mode) as output:
            writer = csv.writer(output)
            if not all([k in already_there.CODE.values for k in keys_to_revise]):
                if mode=='w': writer.writerow(['CODE','N'])
                for key in keys_to_revise:
                    if not key in already_there.CODE.values: 
                        N=len(diags.loc[diags.CIE_CODE==key].PATIENT_ID.unique())
                        writer.writerow([key, N])     
    
    """ READ EVERYTHING """ 
    icd10cm=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str,)
    icd9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD9']), dtype=str,
                     usecols=['ICD-9-CM CODE','CCS LVL 1 LABEL','CCS LVL 2 LABEL',
                              'CCS LVL 3 LABEL','CCS LVL 4 LABEL'])
    diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[yr]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE'],
                      index_col=False)
    
    """ TEXT PREPROCESSING """
    icd10cm=text_preprocessing(icd10cm,
                               [c for c in icd10cm if ((not 'DESCRIPTION' in c) and (not 'LABEL' in c))],
                               mode='icd10cm')
    icd9=text_preprocessing(icd9, [c for c in icd9 if (not 'LABEL' in c)],mode='icd9')
    diags=text_preprocessing(diags, ['CIE_VERSION','CIE_CODE'],mode='diags')
    
    """ ICD 10 CM ASSERTIONS"""
    #Check no null values at reading time
    assert all(icd10cm.isnull().sum()==0), f'Null values encountered when reading {config.ICDTOCCSFILES["ICD10CM"]}'
    assert icd10cm.CCS.isnull().sum()==0, 'Some codes in the ICD10CM dictionary have not been assigned a CCS :('
    
    """ ICD 9 ASSERTIONS"""
    #Check no null values at reading time
    assert all(icd9.isnull().sum()==0), f'Null values encountered when reading {config.ICDTOCCSFILES["ICD9"]}' 
    assert icd9.CCS.describe().isnull().sum()==0, 'Some codes in the ICD9 dictionary have not been assigned a CCS :('

    #%%
    """ DIAGNOSES ASSERTIONS""" 
    #Check null values
    assert all(diags.isnull().sum()==0), f'Null values encountered after cleaning up {config.ICDFILES[yr]}'

    #%%
   
    missing_in_icd9=missingDX(icd9,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('9')])
    missing_in_icd10cm=missingDX(icd10cm,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10')])
    print('Missing quantity ICD9: ', len(missing_in_icd9))
    print('Missing quantity ICD10: ', len(missing_in_icd10cm))
    success9, failure9=guessingCCS(missing_in_icd9, icd9)
    icd9=assign_success(success9,icd9)
    
    f10=os.path.join(config.INDISPENSABLEDATAPATH,f'ccs/manually_revised_icd10.csv')
    assert Path(f10).is_file(), "Manual revision file not found (icd10)!!"
    f9=os.path.join(config.INDISPENSABLEDATAPATH,f'ccs/manually_revised_icd9.csv')
    assert Path(f9).is_file(), "Manual revision file not found (icd9)!!"
    
    revision9=pd.read_csv(f9)
    revision10=pd.read_csv(f10)

    failure9 = {key:val for key, val in failure9.items() if key[0] not in revision9.CODE.values}

    print(f'{len(failure9.keys())} codes need manual revision')
    if len(failure9.keys())>0: 
        needsManualRevision(failure9, icd9,  appendix=f'_icd9', diags=diags)
    print('-------'*10)

    success10, failure10=guessingCCS(missing_in_icd10cm, icd10cm)
    icd10cm=assign_success(success10,icd10cm)
    # keys_to_drop={k[0]:k for k in failure10.keys()}
    failure10 = {key:val for key, val in failure10.items() if key[0] not in revision10.CODE.values} 
    print(f'{len(failure10.keys())} codes need manual revision')
    
    if len(failure10.keys())>0:
        needsManualRevision(failure10, icd10cm, appendix=f'_icd10', diags=diags)
    print('-------'*10)

    revision=pd.concat([revision9,revision10])
    
    #Use the manual revision to change diagnostic codes when necessary
    #Those with no NEW_CODE specified are lost -> discard rows with NAs
    diags2=diags.copy()
    for code, new in zip(revision.CODE, revision.NEW_CODE):
        diags.loc[diags.CIE_CODE==code, 'CIE_CODE']=new
    diags=diags.dropna(subset=['CIE_CODE'])
    L=len(diags)
    print(f'We have lost {len(diags2)-len(diags)} diagnoses that still have no CCS')
    #Keep only IDs present in X, because we need patients to have age, sex and 
    #other potential predictors
    diags=diags.loc[diags.PATIENT_ID.isin(X.PATIENT_ID.values)]
    print(f'We have discarded {L-len(diags)} diagnoses because the patients have no additional predictors such as Age')

    
    #%%
    icd9['DESCRIPTION']=icd9['CCS LVL 1 LABEL']+icd9['CCS LVL 2 LABEL']+icd9['CCS LVL 3 LABEL']+icd9['CCS LVL 4 LABEL']
    icd10cm.rename(columns={'CCS CATEGORY DESCRIPTION':'DESCRIPTION'},inplace=True)
    icd9['CIE_VERSION']='9'
    icd10cm['CIE_VERSION']='10'
    dict9=icd9[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
    dict10=icd10cm[['CODE', 'CCS', 'DESCRIPTION', 'CIE_VERSION']]
    fulldict=pd.concat([dict9,dict10]).drop_duplicates()
    #Drop codes that were not diagnosed to any patient in the current year
    diags_with_ccs=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'CCS':[], 'DESCRIPTION':[]})
    df=diags.copy()
    df['CODE']=df.CIE_CODE.astype(str)
    diags_with_ccs=pd.merge(df, fulldict, on=['CODE','CIE_VERSION'], how='inner')[['PATIENT_ID','CIE_CODE','CODE','CCS']]
    
    #%%
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
    
    X.reindex(sorted(X.columns), axis=1).to_csv(os.path.join(config.DATAPATH,f'CCS{yr}.csv'))
    print('Saved ',os.path.join(config.DATAPATH,f'CCS{yr}.csv'))
    return 0,0
if __name__=='__main__':
    import sys
    sys.path.append('/home/aolza/Desktop/estratificacion/')
    yr=2017
    X,Y=getData(yr, CCS=False)
    # _ , _ =generateCCSData(yr,  X)
    print('positive class ',sum(np.where(Y.urgcms>=1,1,0)))
    
    # xx,yy=CCSData(2016,  X, Y)