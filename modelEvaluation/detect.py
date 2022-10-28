#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:18:31 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import os
import pandas as pd
from pathlib import Path
import re
import argparse

parser = argparse.ArgumentParser(description='Compare models')
parser.add_argument('--year', '-y', type=int,default=argparse.SUPPRESS,
                    help='The year for which you want to compare the predictions.')
parser.add_argument('--nested','-n', dest='nested', action='store_true', default=False,
                    help='Are you comparing nested models with the same algorithm?')
parser.add_argument('--all','-a', dest='all', action='store_true', default=True,
                    help='Compare all models with the same algorithm?')
parser.add_argument('--config_used', type=str, default=argparse.SUPPRESS,
                help='Full path to configuration json file: ')

args, unknown_args = parser.parse_known_args()

from python_settings import settings as config

if not config.configured: 
    import configurations.utility as util
    config_used=args.config_used if hasattr(args, 'config_used') else os.path.join(os.environ['USEDCONFIG_PATH'],input('Experiment...'),input('Model...')+'.json')
    configuration=util.configure(config_used,TRACEBACK=True, VERBOSE=True)
import configurations.utility as util
def detect_models(modelpath=config.MODELPATH):
    available_models=[x.stem for x in Path(modelpath).glob('./*') if x.is_file() and x.suffix in ['.joblib']]
    available_models+=[x.stem for x in Path(modelpath).glob('./neural*') if x.is_dir() or x.suffix in['.joblib', '.pb']] #this can be a file or directory
    print('Available models are:')
    print(available_models)
    return(available_models)

def detect_latest(available_models): 
    if len(available_models)==1:
        print('There is only one model')
        return(available_models)
    print('Selecting latest models per algorithm:')
    available_models=[m+'0' for m in available_models]
    ids = [int(''.join(re.findall('\d+',model))) for model in available_models]
    algorithms=['_'.join(re.findall('[^\d+_\d+]+',model)) for model in available_models]
    df=pd.DataFrame(list(zip(algorithms,ids,[i for i in range(len(ids))])),columns=['algorithm','id','i'])
    selected=df.loc[~(df.algorithm.str.startswith('nested'))].groupby(['algorithm']).apply(lambda x: x.loc[x.id == x.id.max()].i).to_numpy()
    selected=[available_models[i][:-1] for i in selected.ravel()]
    print(selected)
    return(selected)