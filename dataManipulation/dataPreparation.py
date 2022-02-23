#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANOMALIAS: HAY PACIENTES QUE INGRESARON Y NO TIENEN ACGS

len(set(full17.PATIENT_ID)-set(ingT[2017].PATIENT_ID))
Out[78]: 2074286 personas NO ingresaron en 2017
len(set(ingT[2017].PATIENT_ID))
Out[81]: 187335 personas sí ingresaron en 2017

len(set(ingT[2017].PATIENT_ID)-set(full17.PATIENT_ID))
Out[80]: 20345 personas ingresaron pero no tenemos sus ACG!!!!!!

Created on Wed Nov 17 11:20:27 2021

@author: aolza
"""

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
    has_fullEDCs_attr=hasattr(config,'FULLEDCS')
    try:
        fullEDCs = kwargs.get('fullEDCs', config.FULLEDCS)
    except AttributeError:
        fullEDCs = False
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
        return (acg.drop(response,axis=1),acg[['PATIENT_ID',response]])

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

    return(data[finalcols].reindex(sorted(data[finalcols].columns), axis=1),data[cols])

if __name__=='__main__':
    import sys
    sys.path.append('/home/aolza/Desktop/estratificacion/')
    
    # ing=loadIng(config.ALLHOSPITFILE,configs.DATAPATH)
    # ingT=createYearlyDataFrames(ing)
    # x,y=getData(2017)
    # xx,yy=getData(2017,oldbase=True)
    X,Y=getData(2016)
    print('positive class ',sum(np.where(Y.urgcms>=1,1,0)))
    # import inspect
    # used=[createYearlyDataFrames, loadIng,assertMissingCols,
    #       prepare,resourceUsageDataFrames]
    # for  f in used:
    #     print(f.__name__,inspect.signature(f))