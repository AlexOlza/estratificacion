#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:33:56 2022

@author: aolza
"""
from pathlib import Path
import pandas as pd
import time
import re
import numpy as np
from configurations.default import *
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
def load_predictors(path,predictors=None):
    df=pd.read_csv(path,nrows=5)
    if predictors: #nonempty list or str, True boolean, nonzero number 
        if isinstance(predictors,list):
            pass
        elif isinstance(predictors,str):
            print('hello!', predictors)
            predictors=[col for col in 
                    df if bool(re.match(predictors,col))]
            # return df
        else:
            print('str')
            predictors=[p for p in 
                    np.array(df.filter(regex=PREDICTORREGEX).columns)]
    else:
        print('false')
        predictors=[col for col in df]
    if not 'PATIENT_ID' in predictors:
        predictors.insert(0,'PATIENT_ID')
        
    print('predictors: ',predictors)
    # assert False
    return predictors

def create_fullacgfiles(filename,directory=DATAPATH,predictors=None, all_EDCs=True,overwrite=False):
    acg=pd.DataFrame()
    t0=time.time()
    # if Path(directory+FULLACGFILES[year]).is_file():
    #     print(FULLACGFILES[year], ' found: Loading...')
    #     fname=directory+FULLACGFILES[year]
    #     acg=pd.DataFrame()
    #     t0=time.time()

    #     predictors=load_predictors(fname,predictors)
    #     print('Loading ',fname)
    #     for chunk in pd.read_csv(fname, chunksize=100000,usecols=predictors):
    #         d = dict.fromkeys(chunk.select_dtypes(np.int64).columns, np.int8)
    #         d['PATIENT_ID']=np.int64
    #         ignore=[]
    #         for k in d.keys():
    #             if any(np.isnan(chunk[k].values)):
    #                 ignore.append(k)
    #                 for k in ignore:
    #                     d.pop(k)
    #         chunk= chunk.astype(d)
    #         acg = pd.concat([acg, chunk], ignore_index=True)
    #         # break 
    #     print('Loaded in ',time.time()-t0,' seconds')
    #     try:    
    #         acg=acg.drop(labels=['EDC_NUR11','EDC_RES10','RXMG_ZZZX000',
    #                              'ACG_5320','ACG_5330','ACG_5340'],axis=1)
    #         print('dropped')
    #     except KeyError:
    #         print('not dropping cols')
    #     return acg
    
    for path in Path(directory).rglob(filename):
        
        if all_EDCs:
            p=re.sub('\|\|','|',re.sub(r'EDC_|RXMG_|','',predictors))
            print(p)
            edc_data=pd.read_csv(INDISPENSABLEDATAPATH+ALLEDCFILES[year],
            usecols=['patient_id', 'edc_codes', 'rxmg_codes'],
            delimiter=';')
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
            
            if all_EDCs:
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
    acg[sorted(acg, key = lambda x: x not in acg.filter(like="PATIENT_ID").columns)].to_csv(directory+FULLACGFILES[year])
    return(acg)

if __name__=='__main__':
    import sys
    year=int(sys.argv[1])
    filename=ALLEDCFILES[year]
    acgs=create_fullacgfiles(ACGFILES[year],directory=DATAPATH,predictors=PREDICTORREGEX, all_EDCs=True)
    weird=acgs.select_dtypes(include=['float'])
    print('Saved!')