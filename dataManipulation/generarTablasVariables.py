#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTINUACIÓN DE exploratorio/exploratorio.r
GENERA LOS ARCHIVOS usoRecursos2016.csv y usoRecursos2017.csv si no existen en '.'
SI NO, LOS CARGA
DESPUÉS, LOS JUNTA CON LOS ACGS (to do)
Created on Wed Nov 10 15:58:34 2021

@author: aolza

PROBLEMAS:
    1) HAY 11868 PACIENTES SIN INFO DE DIAL-URGENCIAS-CONSULTAS EN 2016 Y 15919 EN 2017
    intu16  has  759  new ids out of  1804
    pxmayor16  has  5271  new ids out of  98070
    cirmayor16  has  5645  new ids out of  108340
    diurco16  has  0  new ids out of  2255111
    cancer16  has  2794  new ids out of  41708
    --------------------------
    intu17  has  707  new ids out of  1640
    pxmayor17  has  5297  new ids out of  99527
    cirmayor17  has  5652  new ids out of  109412
    diurco17  has  0  new ids out of  2256287
    cancer17  has  3004  new ids out of  48043
    
    
    ALL INTERSECTAR CON LOS ACG LOS NA DE DIURCO DESAPARECEN:
        Significa que los datos de diurco fueron extraídos aprox en las mismas
        fechas que los acgs, y el resto de variables más tarde, así que contienen
        más pacientes que los que utilizabamos hasta ahora???

"""
from python_settings import settings as config
import configurations.utility as util
import pandas as pd
import os
import re
from pathlib import Path
import numpy as np
import time

def nonconstant(df,key,nan=True): return (df[key].std()!=0 if nan else (df[key] != df[key].iloc[0]).any())
#PROBLEMA 1:
def intersection(ids,dfs,yr):
    for key in dfs.keys():
        if yr not in key:
            continue
        s=set(dfs[key]['id'])
        intersection=list(set(ids) & s)
        print(key,' has ',len(s)-len(intersection),' new ids out of ',len(s))
        
def multipleMerge(dfyear,dfs):
    for key,df in dfs.items():
        if ('diurco' not in key) :
            util.vprint('joining {0} and {1}...'.format('diurco',key))
            dfyear=pd.merge(dfyear,df,on='id',how='outer')
    for key in dfyear:
        if not nonconstant(dfyear,key):
            dfyear[key]=dfyear[key].fillna(0)
    print(dfyear.info(verbose=config.VERBOSE,null_counts=True))             
    return dfyear

def resourceUsageDataFrames(years=[2016,2017],exploratory=False):
    years=[years] if isinstance(years, int) else years
    dataPath=config.INDISPENSABLEDATAPATH+'datos_hospitalizaciones/'
    filenames=[dataPath+'usoRecursos{0}.csv'.format(yr) for yr in years]
    outputDFs={} #this will be a dict of dataframes
    stryears=[str(yr)[-2:] for yr in years]
    for yr,stryr,f in zip(years,stryears,filenames):
        if not os.path.exists(f):
            print('Generating files')
            
        
            dfs={} #this will be a dict of dataframes
            datos={'intubacion_{0}.rda'.format(yr):'intu{0}'.format(stryr),
                    'procedimiento_mayor grd {0}.rda'.format(yr):'pxmayor{0}'.format(stryr),
                    'cirugia mayor {0}.rda'.format(yr):'cirmayor{0}'.format(stryr),
                    'dial_urgencias_consultas{0}.rda'.format(yr):'diurco{0}'.format(stryr),
                    'tratamiento cancer activo {0}.rda'.format(yr):'cancer{0}'.format(stryr),
                    'residenciado.rda':'resi'}
            try:
                import pyreadr
                for inputFile,df in datos.items():
                    result = pyreadr.read_r(dataPath+inputFile)
                    dfs[df]=result[list(result.keys())[0]].drop_duplicates()
                    del result    
                    
                #Drop constant columns in each dataframe and add a 1 column
                #I am aware of the inefficiency, but It's comfortable for joining later
                for key in dfs.keys():
                    # print(dfs[key].columns)
                    dfs[key]=dfs[key].loc[:, (dfs[key] != dfs[key].iloc[0]).any()] 
                    if 'diurco' not in key: #ill use this as left for joining
                        dfs[key][key]=1
                    # print(dfs[key].columns)
                    # print('\n')
                util.vprint('DataFrames: ')
                for key,df in dfs.items():
                    print(key,list(df.columns),len(df))
                    
                
                if exploratory:        
                    intersection(dfs['diurco{0}'.format(stryr)]['id'],dfs,stryr)
                    # intersection(dfs['diurco17']['id'],dfs,'17')
                
                outputDFs[yr]=dfs['diurco{0}'.format(stryr)]
                outputDFs[yr]=multipleMerge(outputDFs[yr],dfs)
                
                types={'id': 'int64','dial':'Int64','consultas':'Int64',
                       'urgencias':'Int64','resi':'int8'}
                for k in outputDFs[yr].keys():
                    if stryr in k:
                        types[k]='int8'
                
                # outputDFs[2017]=outputDFs[2017].astype(types17)
                outputDFs[yr]=outputDFs[yr].astype(types)
                # for yr,f in zip(years,filenames):
                outputDFs[yr].to_csv(f)
                print('Generated ',f)
            except:
                print('unable to import pyreadr. Cant generate files')
            
         
        else:
            # for f,yr,stryr in zip(filenames,years,stryears):
            outputDFs[yr]=pd.read_csv(f)
            print('Loaded ',f)
            types={'id': 'int64','dial':'Int64','consultas':'Int64',
                   'urgencias':'Int64','resi':'int8'}
            for k in outputDFs[yr].keys():
                if stryr in k:
                    types[k]='int8'
            outputDFs[yr]=outputDFs[yr].astype(types)
    return outputDFs
# #####################################################
#######################################################
"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        JUNTAR CON LOS ACGS
        
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def load_predictors(path,predictors=None):
    df=pd.read_csv(path,nrows=5)
    # print(df)
    if predictors: #nonempty list or str, True boolean, nonzero number 
        if isinstance(predictors,list):
            pass
        elif isinstance(predictors,str):
            predictors=[p for p in 
                    np.array(df.filter(regex=predictors).columns)]
        else:
            predictors=[p for p in 
                    np.array(df.filter(regex=config.PREDICTORREGEX).columns)]
            # predictors.insert(0,'FEMALE')
    else:
        predictors=[col for col in df]
    if not 'PATIENT_ID' in predictors:
        predictors.insert(0,'PATIENT_ID')
        
    util.vprint('predictors: ',predictors)
    # config.PREDICTORS=predictors #NOT SURE IF THIS IS A GOOD IDEA
    return predictors

def load(filename,directory=config.DATAPATH,predictors=None):
	acg=pd.DataFrame()
	t0=time.time()

	for path in Path(directory).rglob(filename):
		predictors=load_predictors(path,predictors)
		# predictors.insert(-1,'ingresoUrg')
		print('Loading ',path)
		for chunk in pd.read_csv(path, chunksize=100000,usecols=predictors):
			d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
			d['PATIENT_ID']=np.int64
			ignore=[]
			for k in d.keys():
				if any(np.isnan(chunk[k].values)):
					ignore.append(k)
 			# print(ignore)
			for k in ignore:
				d.pop(k)
			chunk= chunk.astype(d)
			acg = pd.concat([acg, chunk], ignore_index=True)
		break 
	util.vprint('Loaded in ',time.time()-t0,' seconds')
	try:    
		acg=acg.drop(labels=['EDC_NUR11','EDC_RES10','RXMG_ZZZX000',
		'ACG_5320','ACG_5330','ACG_5340'],axis=1)
		print('dropped')
	except KeyError:
		print('not dropping cols')
		pass
	# constant=[c for c in acg.loc[:,~(acg != acg.iloc[0]).any()].columns]
	# print('Dropping constant columns:',constant)
	# acg.drop(constant,axis=1)
	return(acg)
def retrieveIndicePrivacion(save=True,verbose=config.VERBOSE,yrs=[2016,2017,2018],keep=None,predictors=False):
    keep=list(yrs) if keep is None else ([keep] if isinstance(keep,int) else keep)
    savestr=', save them ' if save else ''
    keepstr=' and return dataframes for years {0}'.format(keep) if keep else ''
    util.vprint('We will look for files from years {0}, generate them if not found{1}{2}'.format(yrs,savestr,keepstr))
    t0=time.time() 
    if keep:
        dfs={yr: pd.DataFrame() for yr in keep}
    for yr in yrs:
        old=config.INDISPENSABLEDATAPATH+'{0}.txt'.format(yr)
        try:
            new=config.DATAPATH+config.ACGFILES[yr]
        except:
            new=config.DATAPATH+config.ACGFILES[str(yr)]
        try:
            f=config.DATAPATH+config.ACGINDPRIVFILES[yr]
        except:
            f=config.DATAPATH+config.ACGINDPRIVFILES[str(yr)]
        if os.path.exists(f):
            util.vprint('found ',f)
            if yr in keep:
                dfs[yr]=load(f.split('/')[-1],predictors=predictors)
                # print(dfs[yr].columns)
                util.vprint(len(dfs[yr][dfs[yr].INDICE_PRIVACION.isnull()]), 'null values for INDICE_PRIVACION')
            continue
        t1=time.time()
        print('Generating ',f)
        indice=pd.read_csv(old,usecols=['PATIENT_ID','INDICE_PRIVACION'],delimiter=';')
        util.vprint('read indice ',time.time()-t1)
        t2=time.time()
        acg=pd.DataFrame()
        for chunk in pd.read_csv(new, chunksize=100000):
                d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
                d['PATIENT_ID']=np.int64
                chunk= chunk.astype(d)
                acg = pd.concat([acg, chunk], ignore_index=True)
        util.vprint('read acg ',time.time()-t2)
        newacg=pd.merge(indice,acg,on='PATIENT_ID')
        if yr in keep:
            dfs[yr]=newacg
        # indice.INDICE_PRIVACION.isnull().any()
        util.vprint(len(newacg[newacg.INDICE_PRIVACION.isnull()]), 'null values for INDICE_PRIVACION')
        if save:
            t3=time.time()
            if '# PATIENT_ID' in newacg.columns:
                newacg.drop('# PATIENT_ID',index=1,inplace=True)
            newacg.to_csv(f,index=False)
            print('Saved ',f,time.time()-t3)
        # break
    util.vprint('time elapsed: ',time.time()-t0)
    if len(keep):
        return([dfs[yr] for yr in keep])
    
def prepare(df16,yr,predictors=False,indicePrivacion=False,verbose=config.VERBOSE):
    if indicePrivacion:
        acg16=retrieveIndicePrivacion(yrs=[yr],keep=yr,verbose=verbose,predictors=predictors)[0]
    else:
        acg16=load(filename=config.ACGFILES[yr],predictors=predictors)
    #delete year reference from variable names for consistency
    newnames={col:re.sub('[1|2][0-9]$','',col) for col in df16}
    newnames['id']='PATIENT_ID'
    df16.rename(columns=newnames,inplace=True)
    full16=pd.merge(acg16,df16,on='PATIENT_ID',how='inner')
    full16=full16.filter(regex=r'^(?!Unnamed).*$',axis=1)
    d=dict.fromkeys(full16.select_dtypes('Int64').columns, 'Int16')
    d['PATIENT_ID']=np.int64
    full16=full16.astype(d)
    del acg16
    return full16
#%% USAGE
if __name__=="__main__":
    d=resourceUsageDataFrames(2017)
    for k in d.keys():
        print(d[k])
        full17=prepare(d[k],2016)


