#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTINUACIÓN DE exploratorio/exploratorio.r
GENERA LOS ARCHIVOS usoRecursos2016.csv y usoRecursos2017.csv si no existen en '.'
SI NO, LOS CARGA
DESPUÉS, LOS JUNTA CON LOS ACGS (to do)
Created on Wed Nov 10 15:58:34 2021

@author: aolza

"""
from python_settings import settings as config
import configurations.utility as util
configuration=util.configure()
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
            outputDFs[yr]=pd.read_csv(f,sep=',|;',engine='python')
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
import csv

def get_delimiter(file_path, bytes = 40960):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter
def load_predictors(path,predictors=config.PREDICTORREGEX):
    df=pd.read_csv(path,nrows=5,sep=get_delimiter(path))
    if predictors: #nonempty list or str
        is_list=isinstance(predictors,list)
        is_str=isinstance(predictors,str)
        assert (is_str or is_list), "Wrong data type of arg predictors. USAGE: load_predictors(path: str or Path, predictors=None: None, str or list)"
        if is_list:
            pass
        elif is_str:
            predictors=[col for col in 
                    df if bool(re.match(predictors,col))]
    else:
        print('Will load full dataset')
        predictors=[col for col in df]
        print(predictors)
    if not 'PATIENT_ID' in predictors:
        predictors.insert(0,'PATIENT_ID')

    return predictors

def load(filename,directory=config.DATAPATH,predictors=config.PREDICTORREGEX):
    acg=pd.DataFrame()
    t0=time.time()
    
    for path in Path(directory).rglob(filename):
        predictors=load_predictors(path,predictors)
        print('Loading ',path)
        for chunk in pd.read_csv(path, chunksize=100000,usecols=predictors,sep=get_delimiter(path)):
            d = dict.fromkeys(chunk.columns, np.int8)
            d['PATIENT_ID']=np.int64
            if 'COSTE_TOTAL_ANO2' in predictors:
                d['COSTE_TOTAL_ANO2']=np.float64
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
                util.vprint(len(dfs[yr][dfs[yr].INDICE_PRIVACION.isnull()]), 'null values for INDICE_PRIVACION')
            continue
        t1=time.time()
        print('Generating ',f)
        indice=pd.read_csv(old,usecols=['PATIENT_ID','INDICE_PRIVACION'],delimiter=',|;',engine='python')
        util.vprint('read indice ',time.time()-t1)
        t2=time.time()
        acg=pd.DataFrame()
        for chunk in pd.read_csv(new, chunksize=100000,sep=',|;',engine='python'):
                d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
                d['PATIENT_ID']=np.int64
                chunk= chunk.astype(d)
                acg = pd.concat([acg, chunk], ignore_index=True)
        util.vprint('read acg ',time.time()-t2)
        newacg=pd.merge(indice,acg,on='PATIENT_ID')
        if yr in keep:
            dfs[yr]=newacg
        util.vprint(len(newacg[newacg.INDICE_PRIVACION.isnull()]), 'null values for INDICE_PRIVACION')
        if save:
            t3=time.time()
            if '# PATIENT_ID' in newacg.columns:
                newacg.drop('# PATIENT_ID',index=1,inplace=True)
            newacg.to_csv(f,index=False)
            print('Saved ',f,time.time()-t3)
    util.vprint('time elapsed: ',time.time()-t0)
    if len(keep):
        return([dfs[yr] for yr in keep])

def create_fullacgfiles(filename,year,directory=config.DATAPATH,
                        predictors=None, all_EDCs=True,overwrite=False):
    if Path(config.DATAPATH+config.FULLACGFILES[year]).is_file():
        if overwrite:
            print(config.FULLACGFILES[year], ' found: Overwriting (may take a while)...')
        else: 
            print(config.FULLACGFILES[year], ' found: Loading...')
            return load(filename=config.FULLACGFILES[year],predictors=predictors)
    acg=pd.DataFrame()
    t0=time.time()
    for path in Path(directory).rglob(filename):
        p=re.sub('\|\|','|',re.sub(r'EDC_|RXMG_|','',predictors))
        print(p)
        edc_data=pd.read_csv(config.INDISPENSABLEDATAPATH+config.ALLEDCFILES[year],
        usecols=['patient_id', 'edc_codes', 'rxmg_codes'],
        delimiter=',|;',engine='python')
        edc_data.rename(columns={'patient_id':'PATIENT_ID'},inplace=True)
        edc_data['all_codes']=edc_data['edc_codes']+' '+edc_data['rxmg_codes']
        
        codes=[]
        for column,prefix in zip(['edc_codes', 'rxmg_codes'],['EDC_','RXMG_']):
            all_codes=[v.split(' ') for v in edc_data[column].values]
            codes+=list(set([prefix+item for sublist in all_codes for item in sublist if item!=''])) #flatten list
        edc_data.drop(columns=['edc_codes', 'rxmg_codes'],axis=1,inplace=True)
        pred=load_predictors(path,p) 
        print('Loading ',path)
        for chunk in pd.read_csv(path, chunksize=100000,usecols=pred):
            d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
            d['PATIENT_ID']=np.int64
            ignore=[]
            for k in d.keys():
                if any(np.isnan(chunk[k].values)):
                    ignore.append(k)
                for k in ignore:
                    d.pop(k)
            chunk= chunk.astype(d)
            data=edc_data.loc[edc_data.PATIENT_ID.isin(chunk.PATIENT_ID.values)]
            
            for code in codes:
                chunk[code]=[int(re.sub('EDC_|RXMG_','',code) in v) for v in data['all_codes'].values]
                chunk[code]=chunk[code].astype(np.int8)
                # print('sum',sum(acg[code].values))
            print('added ',len(codes), 'codes')
        acg = pd.concat([acg, chunk], ignore_index=True)
        break 
    print('Loaded in ',time.time()-t0,' seconds')
    try:    
        acg=acg.drop(labels=['EDC_NUR11','EDC_RES10','RXMG_ZZZX000',
        'ACG_5320','ACG_5330','ACG_5340'],axis=1)
        print('dropped')
    except KeyError:
        print('not dropping cols')
        pass
    acg[sorted(acg, key = lambda x: x not in acg.filter(like="PATIENT_ID").columns)].to_csv(directory+config.FULLACGFILES[year])
    return(acg)
def prepare(df16,yr,predictors=None,indicePrivacion=False,fullEDCs=False,verbose=config.VERBOSE,
            excludeOSI=config.EXCLUDEOSI):
    if indicePrivacion and (not fullEDCs):
        acg16=retrieveIndicePrivacion(yrs=[yr],keep=yr,verbose=verbose,predictors=predictors)[0]
    elif (not indicePrivacion) and (not fullEDCs):
        acg16=load(filename=config.ACGFILES[yr],predictors=predictors)
    elif (not indicePrivacion) and (fullEDCs):
        acg16=create_fullacgfiles(config.FULLACGFILES[yr],yr,directory=config.DATAPATH,
                    predictors=predictors)
    else:
        assert False, 'NOT IMPLEMENTED: indicePrivacion and fullEDCs'
    
    len1=len(acg16)   
    if excludeOSI:
        excludeOSI=[excludeOSI] if isinstance(excludeOSI,str) else excludeOSI
        assert 'osi' in acg16.columns, 'No "osi" column in config.ACGFILES, can not exclude patients from {0} :('.format(excludeOSI)
        acg16=acg16[~acg16['osi'].isin(excludeOSI)] 
        util.vprint('Excluded {0} patients from OSIs {1}'.format(len1-len(acg16),excludeOSI))
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
        full17=prepare(d[k],2016,fullEDCs=(True))


