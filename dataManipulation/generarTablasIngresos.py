#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuación de exploratorio/exploratorio.r
------------------------------------
POTENCIALES VARIABLES RESPUESTA:
    construidas a partir de 'tipo', 'prioridad','planned_cms', 'newborn_injury'
(16 columnas)*2=32 columnas + id = 33
-urg|urg_cms|prog|prog_cms  (solo hospitalizacion, sin quitar newborn_injury)
- hdia_col donde col son urg|urg_cms|prog|prog_cms -solo hosp.dia (para hacer la resta en algunos modelos, sin quitar newborn_injury)
- nbinj_col donde col son urg|urg_cms|prog|prog_cms
- hdia_nbinj_col donde col son urg|urg_cms|prog|prog_cms
------------------------------------

Created on Fri Nov 12 12:01:39 2021

@author: aolza
"""
from pathlib import Path
# from datetime import timedelta
from warnings import warn
import pandas as pd
import numpy as np
import time
# from configurations import config
from python_settings import settings as config
import configurations.utility as util
def assertMissingCols(needed,df,function_name):
    missing_columns=set(needed)-set(df.columns)
    assert len(missing_columns)==0,'{1}: MISSING NEEDED COLS {0} IN THE HOSPITALIZATION DATASET.'.format(missing_columns,function_name)
    
def loadIng(f='ingresos2016_2018.csv',project_dir=config.DATAPATH):
    for path in Path(project_dir).rglob(f):
        ing=pd.read_csv(path)
        break  
    return ing
#%%
def createYearlyDataFrames(ing):
     #Assertions and warnings:
    assertMissingCols(['fecing'], ing, 'createYearlyDataFrames')
    """Returns a dictionary of DFs with years as keys, containing the 
    hospitalization info for each patient id in each year"""
    ing.fecing=pd.to_datetime(ing.fecing)
    years=ing.fecing.dt.year.unique()
    dic=dict()
    util.vprint('Hospital admissions in years (keys of returned dict): ',years)
    for y in years:
        df=(ing.loc[ing.fecing.dt.year==y]).copy(deep=True)
        dic[y]=transform(df)
    return(dic)

def transform(df,verbose=True):  #all of this can be written more compactly with loops!!!!! 
    t0=time.time()
    #Assertions and warnings:
    needed=['id', 'tipo','prioridad','planned_cms', 'newborn_injury']
    assertMissingCols(needed, df, 'transform')
    if not set(['URGENTE']).issubset(df.prioridad.unique()):
        warn('THERE ARE NO ROWS WITH prioridad=="URGENTE". Is this expected? Review function transform.')
    if not set(['h.dia']).issubset(df.tipo.unique()):
        warn('THERE ARE NO ROWS WITH tipo=="h.dia". Is this expected? Review function transform.')
    if not 'ing' in df.columns:
        df['ing']=1
        
    #Characteristics of each hospitalization (boolean arrays)
    urg=(df.prioridad=='URGENTE')
    plancms=(df.planned_cms==1)
    hdia=(df.tipo=='h.dia')
    nbinj=(df.newborn_injury==1)
    #auxiliary cols
    df['nbinj']=df['newborn_injury']#shorter name
    df['tot']=df.groupby(['id']).ing.transform(sum)
    #Describing all the characteristics of each episode
    #And computing the number of similar episodes per id using groupby and transform(sum)
    ###################################################
    df['urg']=np.where((~hdia) & (urg), 1,0)
    df['prog']=np.where((~hdia) & (~urg), 1,0)
    df['urg_num']=df.groupby(['id'])['urg'].transform(sum)
    df['prog_num']=df.groupby(['id'])['prog'].transform(sum)
    
    df['urgcms']=np.where((~hdia) & (~plancms), 1,0)
    df['progcms']=np.where((~hdia) & (plancms), 1,0)
    df['urgcms_num']=df.groupby(['id'])['urgcms'].transform(sum)
    df['progcms_num']=df.groupby(['id'])['progcms'].transform(sum)
    
    df['nbinj_urg']=np.where((urg) & (nbinj),1,0)
    df['nbinj_prog']=np.where((~urg) & (nbinj),1,0)
    df['nbinj_urgcms']=np.where((~plancms) & (nbinj),1,0)
    df['nbinj_progcms']=np.where((plancms) & (nbinj),1,0)
    df['nbinj_urg_num']=df.groupby(['id'])['nbinj_urg'].transform(sum)
    df['nbinj_prog_num']=df.groupby(['id'])['nbinj_prog'].transform(sum)
    df['nbinj_urgcms_num']=df.groupby(['id'])['nbinj_urgcms'].transform(sum)
    df['nbinj_progcms_num']=df.groupby(['id'])['nbinj_progcms'].transform(sum)
    
    df['hdia_urg']=np.where((hdia) & (urg), 1,0)
    df['hdia_prog']=np.where((hdia) & (~urg), 1,0)
    df['hdia_urg_num']=df.groupby(['id'])['hdia_urg'].transform(sum)
    df['hdia_prog_num']=df.groupby(['id'])['hdia_prog'].transform(sum)
    df['hdia_urgcms']=np.where((hdia) & (~plancms), 1,0)
    df['hdia_progcms']=np.where((hdia) & (plancms), 1,0)
    df['hdia_urgcms_num']=df.groupby(['id'])['hdia_urgcms'].transform(sum)
    df['hdia_progcms_num']=df.groupby(['id'])['hdia_progcms'].transform(sum)
    
    df['hdia_nbinj_urg']=np.where((hdia) & (urg) & (nbinj),1,0)
    df['hdia_nbinj_prog']=np.where((hdia) & (~urg) & (nbinj),1,0)
    df['hdia_nbinj_urgcms']=np.where((hdia) & (~plancms) & (nbinj),1,0)
    df['hdia_nbinj_progcms']=np.where((hdia) & (plancms) & (nbinj),1,0)
    df['hdia_nbinj_urg_num']=df.groupby(['id'])['hdia_nbinj_urg'].transform(sum)
    df['hdia_nbinj_prog_num']=df.groupby(['id'])['hdia_nbinj_prog'].transform(sum)
    df['hdia_nbinj_urgcms_num']=df.groupby(['id'])['hdia_nbinj_urgcms'].transform(sum)
    df['hdia_nbinj_progcms_num']=df.groupby(['id'])['hdia_nbinj_progcms'].transform(sum)

    cols=[col for col in df if col.endswith('_num')]
    cols.insert(0,'id')
    dd=pd.DataFrame()
    dd=df[cols].drop_duplicates()
    dd.rename(columns=lambda x: x.replace('_num',''), inplace=True)
    types={k:'int16' for k in dd.columns}
    types['id']='int64'
    dd=dd.astype(types)
    dd.rename(columns={'id':'PATIENT_ID'},inplace=True)
    util.vprint('transform time ',time.time()-t0)
    return dd

#%% USAGE
if __name__=="__main__":
#%%
    # print('imported')
    ing=loadIng()
    ingT=createYearlyDataFrames(ing)
    
    #%%
    for y in ingT.keys():
        print(len(ingT[y]),'patients had admissions in ',y)
        # print(ingT[y].describe())
        print('constant cols:',[c for c in ingT[y].loc[:,~(ingT[y] != ingT[y].iloc[0]).any()].columns])
        print('\n')

