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
    elif CCS:
        full16=generateCCSData(yr,  pd.DataFrame(), predictors=predictors)
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
    predictors=kwargs.get('predictors',None)
    filename=os.path.join(config.DATAPATH,config.CCSFILES[yr])
    if Path(filename).is_file():
        return load(config.CCSFILES[yr],directory=config.DATAPATH,
                    predictors=predictors)
    def missingDX(dic,diags):
        diagsCodes=diags.CIE_CODE.str.replace(r'\s|\/', r'').values
        dictCodes=dic.CODE.astype(str).str.replace(r'\s|\/', r'').values
        diagsCodes=set(diagsCodes)
        dictCodes=set(dictCodes)
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
 
            if '259' in options: options.remove('259') #CCS 259 is for residual unclassified codes
            if len(options)==1:
                success[(dx, code)]=options[0]
            else:
                failure[(dx, code)]=options
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
                            description=True, diags=None,
                            exclude={}):      
        import csv 
        with open(f'needs_manual_revision{appendix}.csv', 'w') as output:
            writer = csv.writer(output)
            for key, value in failure.items():
                if key not in exclude.keys():
                    if description:
                        writer.writerow([key[0]+' -> '+key[1],'CCS', 'Description'])
                        for v in value:
                            writer.writerow([' ',v, dictionary.loc[dictionary.CCS==v]['MULTI CCS LVL 2 LABEL'].unique()[0]])
                        writer.writerow(['','',''])
                    else:
                        assert isinstance(diags, pd.DataFrame)
                        N=len(diags.loc[diags.CIE_CODE==key[0]].PATIENT_ID.unique())
                        writer.writerow([key[0], N])     
   
    icd10cm=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD10CM']),
                        dtype=str,)# usecols=['ICD-10-CM CODE', 'CCS CATEGORY'])
    icd10cm.rename(columns={'ICD-10-CM CODE':'CODE', 'CCS CATEGORY':'CCS'},inplace=True)
    icd10cm.CODE=icd10cm.CODE.str.slice(0,6)
    assert icd10cm.CCS.isnull().sum()==0, 'Some codes in the ICD10CM dictionary have not been assigned a CCS :('
    icd9=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDTOCCSFILES['ICD9']), dtype=str,
                     usecols=['ICD-9-CM CODE','CCS LVL 1 LABEL','CCS LVL 2 LABEL',
                              'CCS LVL 3 LABEL','CCS LVL 4 LABEL'])
    

    #the CCS category is expressed between brackets and followed by a dot,
    #and can be inside any column of type 'CCS LVL x LABEL'.
    icd9.rename(columns={'ICD-9-CM CODE':'CODE'},inplace=True)
    icd9['CCS']=icd9['CCS LVL 1 LABEL']+icd9['CCS LVL 2 LABEL']+icd9['CCS LVL 3 LABEL']+icd9['CCS LVL 4 LABEL']
    icd9.CCS=icd9.CCS.str.extract(r'(?P<CCS>[0-9]+)').CCS
    assert icd9.CCS.describe().isnull().sum()==0, 'Some codes in the ICD9 dictionary have not been assigned a CCS :('
    # icd9.CCS=icd9.CCS.str.replace(r'[^0-9]', r'') #Keep only numbers (the CCS category)
    icd9.CCS=icd9.CCS.str.replace(r'\s|\/', r'')
    icd9.CODE=icd9.CODE.str.slice(0,5)
    
    diags=pd.read_csv(os.path.join(config.INDISPENSABLEDATAPATH,config.ICDFILES[yr]),
                      usecols=['PATIENT_ID','CIE_VERSION','CIE_CODE','START_DATE','END_DATE'],
                      index_col=False)
    #KEEP ONLY DX THAT ARE STILL ACTIVE AT THE BEGINNING OF THE CURRENT YEAR
    diags=diags.loc[diags.END_DATE>=f'{yr}-01-01'][['PATIENT_ID', 'CIE_VERSION', 'CIE_CODE']]
    diags.CIE_CODE=diags.CIE_CODE.astype(str)
    diags.CIE_VERSION=diags.CIE_VERSION.astype(str)
    diags.CIE_CODE=diags.CIE_CODE.str.replace(r'\s|\/', r'')
    
    #In the diagnoses dataset, ICD10CM dx that start with a digit are related to oncology
    diags.loc[(diags.CIE_VERSION.astype(str).str.startswith('10') & diags.CIE_CODE.str.match('^[0-9]')),'CIE_CODE']='ONCOLOGY'
    
    #ICD10CM and ICD9 only allow for 6 and 5 characters respectively
    diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10'),'CIE_CODE']=diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10'),'CIE_CODE'].str.slice(0,6)
    diags.loc[diags.CIE_VERSION.astype(str).str.startswith('9'),'CIE_CODE']=diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10'),'CIE_CODE'].str.slice(0,5)
    
   
    missing_in_icd9=missingDX(icd9,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('9')])
    missing_in_icd10cm=missingDX(icd10cm,diags.loc[diags.CIE_VERSION.astype(str).str.startswith('10')])
    print('ICD9 CODES PRESENT IN DIAGNOSTIC DATASET BUT MISSING IN THE DICTIONARY:')
    print('Quantity: ', len(missing_in_icd9))
    success, failure=guessingCCS(missing_in_icd9, icd9)
    icd9=assign_success(success,icd9)
    print(f'{len(failure.keys())} codes need manual revision')
    if len(failure.keys())>0: 
        needsManualRevision(failure, icd9, description=False, appendix=f'nodesc_icd9_{yr}', diags=diags)
    print('-------'*10)
    print('ICD10 CODES PRESENT IN DIAGNOSTIC DATASET BUT MISSING IN THE DICTIONARY:')
    print('Quantity: ', len(missing_in_icd10cm))
    success, failure=guessingCCS(missing_in_icd10cm, icd10cm)
    icd10cm=assign_success(success,icd10cm)
    print(f'{len(failure.keys())} codes need manual revision')
    if len(failure.keys())>0:
        needsManualRevision(failure, icd10cm, appendix=f'_icd10_{yr}', diags=diags)
    print('-------'*10)
    
    f=os.path.join(config.INDISPENSABLEDATAPATH,f'ccs/manually_revised_icd10_{yr}.csv')
    assert Path(f).is_file(), "Manual revision file not found!!"
    revision=pd.read_csv(f, header=0, names=['CODE', 'CCS', 'NEW_CODE'])
    
    #Use the manual revision to change diagnostic codes when necessary
    #Those with no NEW_CODE specified are lost -> discard rows with NAs
    #the CCSs are not correct, look at dictionary -> discard column
    revision=revision.dropna()[['CODE','NEW_CODE']] 
    for code, new in zip(revision.CODE, revision.NEW_CODE):
        diags.loc[diags.CIE_CODE==code, 'CIE_CODE']=new
        
    
    # Add columns of zeros for each CCS category
    for ccs_number in sorted(list(['0'])+list(set(list(icd9.CCS.unique()) + list(icd10cm.CCS.unique())))):
        X[f'CCS{ccs_number}']=np.int16(0)
    
    # Para cada paciente, seleccionar todos sus diagnosticos (icd9_id union icd10_id)
    # Para cada diagnóstico de cada paciente, buscar la categoria CCS en la tabla correspondiente 
    # y sumar 1 a dicha categoría en X

    #Keep only IDs present in X, because we need patients to have age, sex and 
    #other potential predictors
    diags=diags.loc[diags.PATIENT_ID.isin(X.PATIENT_ID.values)]
    icd9diags=diags.CIE_VERSION.astype(str).str.startswith('9')
    icd10diags=diags.CIE_VERSION.astype(str).str.startswith('10')
    
    assert diags.loc[icd9diags].CIE_CODE.isnull().sum()==0
    assert diags.loc[icd10diags].CIE_CODE.isnull().sum()==0
    
    
    diags_with_ccs=pd.DataFrame()
    for version, dictdf in zip( [icd9diags,icd10diags], [icd9, icd10cm]):
        df=diags.loc[version]
        df['CODE']=df.CIE_CODE
        dfmerged=pd.merge(df, icd10cm, how='left', on='CODE')[['PATIENT_ID','CIE_CODE','CODE','CCS', 'CCS CATEGORY DESCRIPTION']]
        diags_with_ccs= pd.concat([diags_with_ccs, dfmerged])      
    
    # Add columns of zeros for each CCS category
    for ccs_number in sorted(list(['0'])+list(set(list(icd9.CCS.unique()) + list(icd10cm.CCS.unique())))):
        diags_with_ccs[f'CCS{ccs_number}']=np.int16(0)
    i=0
    import time
    t0=time.time()
    # X.set_index('PATIENT_ID', inplace=True)
    for ccs_number, df in diags_with_ccs.groupby('CCS'):
        # if i>1000:
        #     break
            # ids=df.PATIENT_ID.unique()
        print(ccs_number,df['CCS CATEGORY DESCRIPTION'].values[0])
        amount_per_patient=df.groupby('PATIENT_ID').size().to_frame(name=f'CCS{ccs_number}')
        X[f'CCS{ccs_number}']=np.int16(0)
        
        X.update(amount_per_patient)

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
    yr=2016
    X,Y=getData(yr)
    # _ , _ =generateCCSData(yr,  X)
    print('positive class ',sum(np.where(Y.urgcms>=1,1,0)))
    
    # xx,yy=CCSData(2016,  X, Y)