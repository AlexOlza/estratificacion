#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:33:56 2022

@author: aolza
"""

import pandas as pd
import time
import re
import numpy as np
# from python_settings import settings as config

# import configurations.utility as util
# configuration=util.configure()
# from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng,assertMissingCols, report_transform
# from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load
# from matplotlib_venn import venn2

DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
INDISPENSABLEDATAPATH=DATAPATH+'indispensable/'

"""FILES"""
ALLHOSPITFILE='ingresos2016_2018.csv'
ACGFILES={2016:'ing2016-2017Activos.csv',
          2017:'ing2017-2018Activos.csv',
          2018:'ing2018-2019Activos.csv'}
ACGINDPRIVFILES={2016:'ing2016-2017ActivosIndPriv.csv',
                  2017:'ing2017-2018ActivosIndPriv.csv',
                  2018:'ing2018-2019ActivosIndPriv.csv'}
ALLEDCFILES={2016:'additional_edcs2016.csv',
             2017:'additional_edcs2017.csv'}
FULLACGFILES={2016:'fullacgs2016.csv',
             2017:'fullacgs2017.csv'}
PREDICTORREGEX=r'PATIENT_ID|FEMALE|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT'
year=2016

from pathlib import Path
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

def load(filename,directory=DATAPATH,predictors=None, all_EDCs=True):
    acg=pd.DataFrame()
    t0=time.time()
    
    for path in Path(directory).rglob(filename):
        
        if all_EDCs:
            p=re.sub('\|\|','|',re.sub(r'EDC_|RXMG_|','',predictors))
            print(p)
            edc_data=pd.read_csv(INDISPENSABLEDATAPATH+ALLEDCFILES[year],
            usecols=['patient_id', 'edc_codes', 'rxmg_codes'],
            delimiter=';')
            edc_data.rename(columns={'patient_id':'PATIENT_ID'},inplace=True)
            types={c:np.int8 for c in edc_data}
            types['PATIENT_ID']=np.int64
            edc_data=edc_data.astype(types)
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
            acg = pd.concat([acg, chunk], ignore_index=True)
            if all_EDCs:
                data=edc_data.loc[edc_data.PATIENT_ID.isin(chunk.PATIENT_ID.values)]
                for column,prefix in zip(['edc_codes', 'rxmg_codes'],['EDC_','RXMG_']):
                    all_codes=[v.split(' ') for v in data[column].values]
                    all_codes=[item for sublist in all_codes for item in sublist if item!=''] #flatten list
                    for code in set(all_codes):
                        acg[prefix+code]=np.where(code in data[column],1,0)
                    print('added ',len(set(all_codes)), 'codes')
        break 
    print('Loaded in ',time.time()-t0,' seconds')
    try:    
        acg=acg.drop(labels=['EDC_NUR11','EDC_RES10','RXMG_ZZZX000',
        'ACG_5320','ACG_5330','ACG_5340'],axis=1)
        print('dropped')
    except KeyError:
        print('not dropping cols')
        pass
    return(acg)

year=2016
filename=ALLEDCFILES[year]
acgs=load(ACGFILES[2016],directory=DATAPATH,predictors=PREDICTORREGEX, all_EDCs=True)
acgs.to_csv(FULLACGFILES[2016])