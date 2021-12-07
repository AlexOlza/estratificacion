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
from python_settings import settings as config
assert config.configured, 'CONFIGURE FIRST!'
import configurations.utility as util
from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng,assertMissingCols
from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load
# from matplotlib_venn import venn2
def listIntersection(a,b): return list(set(a) & set(b))
# FIXME not robust!!!
def excludeHosp(df,filtro,criterio):
    remove='{0}_{1}'.format(filtro,criterio)
    assertMissingCols([remove,criterio],df,'exclude')
    df[criterio]=df[criterio]-df[remove]

# TODO MOVE ASSERTIONS BEFORE LOADING BIG FILES!!!!
# TODO TEST NEW DEFAULT predictors=config.PREDICTORREGEX INSTEAD OF predictors=config.PREDICTORS
# config.PREDICTORS may be currently unused, consider removing it
#OLDBASE IS OBSOLETE
def getData(yr,columns=config.COLUMNS,previousHosp=config.PREVIOUSHOSP,predictors=config.PREDICTORREGEX,
            exclude=config.EXCLUDE,
            resourceUsage=config.RESOURCEUSAGE,
            **kwargs):
    oldbase = kwargs.get('oldbase','OLDBASE' in config.CONFIGNAME)
    if oldbase:
        acg=load(filename=config.ACGFILES[yr],predictors=predictors)
        print('not opening allhospfile')
        return (acg.drop('ingresoUrg',axis=1),acg[['PATIENT_ID','ingresoUrg']])
    cols=columns.copy() #FIXME find better approach
    t0=time.time()
    ing=loadIng(config.ALLHOSPITFILE,config.ROOTPATH)
    ingT=createYearlyDataFrames(ing)
    missing_columns=set([yr,yr+1])-set(ingT.keys())
    assert len(missing_columns)==0,'getData: HOSPITALIZATION DATA FOR YEAR {0} NOT AVAILABLE.'.format(missing_columns)
    del ing
    if exclude:
        assert isinstance(exclude,(str,list)), 'getData accepts str or list or None/False as exclude!'
        exclude=[exclude] if isinstance(exclude,str) else exclude
        for filtro in exclude:
            for c in columns:
                excludeHosp(ingT[yr+1], filtro, c)
        util.vprint('excluded ',exclude)
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
        pred16=pd.merge(full16,ingT[yr]['PATIENT_ID'],on='PATIENT_ID',how='left')

            
    y17=pd.merge(ingT[yr+1],full16['PATIENT_ID'],on='PATIENT_ID',how='outer').fillna(0)
    data=pd.merge(pred16,y17,on='PATIENT_ID')
    print('getData time: ',time.time()-t0)
    return(data[listIntersection(data.columns,pred16.columns)],data[cols])

if __name__=='__main__':
    import sys
    sys.path.append('/home/aolza/Desktop/estratificacion/')
    ing=loadIng(config.ALLHOSPITFILE,config.ROOTPATH)
    ingT=createYearlyDataFrames(ing)

    # import inspect
    # used=[createYearlyDataFrames, loadIng,assertMissingCols,
    #       prepare,resourceUsageDataFrames]
    # for  f in used:
    #     print(f.__name__,inspect.signature(f))