#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:21:11 2021

@author: aolza
"""

def resourceUsageDataFrames(exploratory=False,years=[2016,2017]):
    dataPath=config.dataPath+'datos_hospitalizaciones/'
    filenames=[dataPath+'usoRecursos{0}.csv'.format(yr) for yr in years]
    outputDFs={} #this will be a dict of dataframes
    if not all([os.path.exists(f) for f in filenames]):
        print('Generating files')
        dfs={} #this will be a dict of dataframes
        datos={'intubacion_2016.rda':'intu16','intubacion_2017.rda':'intu17',
                    'procedimiento_mayor grd 2016.rda':'pxmayor16',
                    'procedimiento_mayor grd 2017.rda':'pxmayor17',
                    'cirugia mayor 2016.rda':'cirmayor16','cirugia mayor 2017.rda':'cirmayor17',
                    'dial_urgencias_consultas2016.rda':'diurco16',
                    'dial_urgencias_consultas2017.rda':'diurco17',
                    'tratamiento cancer activo 2016.rda':'cancer16',
                    'tratamiento cancer activo 2017.rda':'cancer17',
                    'residenciado.rda':'resi'}
        for f,df in datos.items():
            result = pyreadr.read_r(dataPath+f)
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
        config.vprint('DataFrames: ')
        for key,df in dfs.items():
            print(key,list(df.columns),len(df))
            
        
        if exploratory:        
            intersection(dfs['diurco16']['id'],dfs,'16')
            intersection(dfs['diurco17']['id'],dfs,'17')
        
        outputDFs[2016],outputDFs[2017]=dfs['diurco16'],dfs['diurco17'] 
        outputDFs[2016]=multipleMerge(outputDFs[2016],dfs,'17')
        outputDFs[2017]=multipleMerge(outputDFs[2017],dfs,'16')      
        
        
        types={'id': 'int64','dial':'Int64','consultas':'Int64',
               'urgencias':'Int64','resi':'int8'}
        types16,types17={key:val for key,val in types.items()},{key:val for key,val in types.items()}
        for k in outputDFs[2016].keys():
            if '16' in k:
                types16[k]='int8'
        for k in outputDFs[2017].keys():
            if '17' in k:
                types17[k]='int8'
        outputDFs[2017]=outputDFs[2017].astype(types17)
        outputDFs[2016]=outputDFs[2016].astype(types16)
        for yr,f in zip(years,filenames):
            outputDFs[yr].to_csv(f)
            print('Generated ',f)
     
    else:
        for f,yr in zip(filenames,years):
            outputDFs[yr]=pd.read_csv(f)
            print('Loaded ',f)
            types={'id': 'int64','dial':'Int64','consultas':'Int64',
                   'urgencias':'Int64','resi':'int8'}
            for k in outputDFs[yr].keys():
                if str(yr)[-2:] in k:
                    types[k]='int8'
            outputDFs[yr]=outputDFs[yr].astype(types)
    return outputDFs