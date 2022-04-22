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
from python_settings import settings as config

import configurations.utility as util
configuration=util.configure()
from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng,assertMissingCols, report_transform
from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load, create_fullacgfiles
# from matplotlib_venn import venn2
def listIntersection(a,b): return list(set(a) & set(b))
# FIXME not robust!!!
def excludeHosp(df,filtros,criterio):#FIXME the bug is here
    filtros=['{0}_{1}'.format(f,criterio) for f in filtros]

    assertMissingCols(filtros+[criterio],df,'exclude')
    
    anyfiltros=(df[filtros].sum(axis=1)>=1)
    crit=(df[criterio]>=1)
    
    df['remove']=np.where((anyfiltros & crit),1,0)

    df[criterio]=df[criterio]-df['remove']

    
# TODO MOVE ASSERTIONS BEFORE LOADING BIG FILES!!!!
#OLDBASE IS OBSOLETE
def getData(yr,columns=config.COLUMNS,previousHosp=config.PREVIOUSHOSP,
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

    cols=columns.copy() #FIXME find better approach
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
    else:
        full16=load(filename=config.ACGFILES[yr],predictors=predictors)
   
    if previousHosp:
        previousHosp=[previousHosp] if isinstance(previousHosp,str) else previousHosp
        assert isinstance(previousHosp, list), 'getData accepts str or list or None as previousHosp!'
        if 'PATIENT_ID' not in previousHosp: 
            previousHosp.insert(0,'PATIENT_ID')
        assertMissingCols(previousHosp,ingT[yr],'getData')
        pred16=pd.merge(full16,ingT[yr][previousHosp],on='PATIENT_ID',how='left').fillna({c:0 for c in previousHosp},inplace=True)
    else:
        assert 'PATIENT_ID' in full16.columns
        pred16=pd.merge(full16,ingT[yr]['PATIENT_ID'],on='PATIENT_ID',how='left')

            
    y17=pd.merge(ingT[yr+1],full16['PATIENT_ID'],on='PATIENT_ID',how='outer').fillna(0)
    print('number of patients y17: ', sum(np.where(y17[config.COLUMNS]>=1,1,0)))
    data=pd.merge(pred16,y17,on='PATIENT_ID')
    print('number of patients in data: ', sum(np.where(data[config.COLUMNS]>=1,1,0)))
    print('getData time: ',time.time()-t0)
    finalcols=listIntersection(data.columns,pred16.columns)
    X,y=data[finalcols].reindex(sorted(data[finalcols].columns), axis=1),data[cols]
    # if CCS:
    #     X,y=CCSData(yr, X, y, **kwargs)
    return(X[finalcols].reindex(sorted(data[finalcols].columns), axis=1),y[cols])

def generateCCSData(yr,  X,
            **kwargs):
    icd10cm=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str,)# usecols=['ICD-10-CM CODE', 'CCS CATEGORY'])
    icd10cm.rename(columns={'ICD-10-CM CODE':'CODE', 'CCS CATEGORY':'CCS'},inplace=True)
    
    #IDENTIFY MISSING CCS CATEGORIES
    print('CCS categories missing in the dictionary: ',set(range(260))-set(icd10cm.CCS.values.astype(int)))
    #IDENTIFY EXTRA CATEGORIES (NOT PRESENT IN REFERENCE WEBSITE)
    # icd10cm.loc[(icd10cm.CCS).astype(int)>259][['CCS', 'CCS CATEGORY DESCRIPTION']].drop_duplicates()
    
    icd9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD9']), dtype=str,
                     )

    
    icd9.rename(columns={'ICD-9-CM CODE':'CODE', 'CCS LVL 1':'CCS'},inplace=True)
    
    diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[yr]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE','START_DATE','END_DATE'],
                      index_col=False)
    #KEEP ONLY DX THAT ARE STILL ACTIVE AT THE BEGINNING OF THE CURRENT YEAR
    diags=diags.loc[diags.END_DATE>=f'{yr}-01-01']

    print('ICD10CM')
    for c in icd10cm:
        print(f'{c} has {len(icd10cm[c].unique())} unique values')
    print(' ')
    print('ICD9')
    for c in icd9:
        print(f'{c} has {len(icd9[c].unique())} unique values')
           
    # Add columns of zeros for each CCS category
    for ccs_number in sorted(list(set(list(icd9.CCS.unique()) + list(icd10cm.CCS.unique())))):
        X[f'CCS{ccs_number}']=np.int16(0)
    
    # Para cada paciente, seleccionar todos sus diagnosticos (icd9_id union icd10_id)
    # Para cada diagnóstico de cada paciente, buscar la categoria CCS en la tabla correspondiente 
    # y sumar 1 a dicha categoría en X
    missing_in_icd9, missing_in_icd10cm = set(),set()

    #Keep only IDs present in X, because we need patients to have age, sex and 
    #other potential predictors
    diags=diags.loc[diags.PATIENT_ID.isin(X.PATIENT_ID.values)]
    i=0
    for _, df in diags.groupby('PATIENT_ID'): 
        id=df.PATIENT_ID.values[0]
        # print(id)
        # df=diags.loc[diags.PATIENT_ID==id]
        icd9_id=df[df.CIE_VERSION.astype(str).str.startswith('9')]
        icd10cm_id=df[df.CIE_VERSION.astype(str).str.startswith('10')]
        for code in icd9_id.CIE_CODE.values:
            if code in icd9.CODE.values:
                ccs_number=icd9[icd9.CODE==code].CCS.values[0]
                X.loc[X.PATIENT_ID==id, f'CCS{ccs_number}']+=np.int16(1)
            else:
                missing_in_icd9.add(code)
        for code in icd10cm_id.CIE_CODE.values:
            if code in icd10cm.CODE.values:
                ccs_number=icd10cm[icd10cm.CODE==code].CCS.values[0]
                X.loc[X.PATIENT_ID==id, f'CCS{ccs_number}']+=np.int16(1)
            else:
                missing_in_icd10cm.add(code)
        i+=1
    print(f'{i} patients processed')
    print('ICD9 CODES PRESENT IN DIAGNOSTIC DATASET BUT MISSING IN THE DICTIONARY:')
    print(missing_in_icd9)
    print('-------'*10)
    print('ICD10 CODES PRESENT IN DIAGNOSTIC DATASET BUT MISSING IN THE DICTIONARY:')
    print(missing_in_icd10cm)
    print('-------'*10)
    
    X.reindex(sorted(X.columns), axis=1).to_csv(os.path.join(config.DATAPATH,f'CCS{yr}.csv'))
    print('Saved ',os.path.join(config.DATAPATH,f'CCS{yr}.csv'))
    return 0,0
if __name__=='__main__':
    import sys
    sys.path.append('/home/aolza/Desktop/estratificacion/')
    yr=2016
    X,Y=getData(yr)
    _ , _ =generateCCSData(yr,  X)
    print('positive class ',sum(np.where(Y.urgcms>=1,1,0)))
    
    # xx,yy=CCSData(2016,  X, Y)