#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:33:34 2021

@author: aolza
"""
# from python_settings import settings as config
import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from joblib import dump
from datetime import datetime as dt
from configurations import utility as util
import re

def readonly(filename):  os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)

from stat import S_IWUSR # Need to add this import to the ones above

def readwrite(filename): os.chmod(filename, S_IWUSR|S_IREAD) # This makes the file read/write for the owner

    
def savemodel(config,model,**kwargs):
    timestamp = str(dt.now())[:19]
    timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars
    NOW = re.sub(r'[\s]','_', timestamp)
    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, 'relative/path/to/file/you/want')
    if not os.path.exists(config.MODELPATH):
        os.mkdir(config.MODELPATH)
    modelname=config.ALGORITHM+NOW
    modelfilename=modelname+'.joblib'
    configname=config.USEDCONFIGPATH+modelname+'.json'
    os.chdir(config.MODELPATH)
    print('dump',config.MODELPATH+modelfilename)
    dump(model, config.MODELPATH+modelfilename)
    os.chdir(config.USEDCONFIGPATH)
    attrs=saveconfig(config,configname)
    util.info(modelfilename,conf=attrs,**kwargs)
import json
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def saveconfig(config,configname):
    settings=config.__dict__['_wrapped'].default_settings.__dict__
    attrs={k:v for k,v in settings.items() if k==k.upper()}#POTENTIALLY INCLUDE MORE FIELDS
    for k,v in attrs.items():
        if not is_jsonable(v):
            attrs[k]=str(v)
    with open(configname,'w') as f:
        json.dump(attrs, f)
    return attrs
    
        
            