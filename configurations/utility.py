#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" UTILITY FUNCTIONS THAT HAVE TO DO WITH SETTINGS 
    should be imported as: from configurations import utility as util
"""
    
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
from joblib import dump
from configurations import utility as util
import re
import json
import os
from datetime import datetime as dt
from contextlib import redirect_stdout
from python_settings import settings as config #MUST BE ALREADY CONFIGURED

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

vprint = print if config.VERBOSE else lambda *a, **k: None
 
def makeAllPaths():
    settings=config.__dict__['_wrapped'].default_settings.__dict__
    paths={k:v for k,v in settings.items() if 'PATH' in k}
    for p in paths.values():
        if not os.path.exists(p):
            os.makedirs(p)
            vprint('created directory ',p)
    vprint('All paths exist now')
        
""" INFO FILE"""
def info(modelfilename,modelPath=config.MODELSPATH,**kwargs):
    now = dt.now() # current date and time
    file=os.path.join(modelPath,'available_models.txt')
    with open(file, 'a+') as f:
        with redirect_stdout(f):
            print('Experiment date: ',now.strftime("%m/%d/%Y, %H:%M:%S"))
            print('Model filename: ',modelfilename)
            for arg,val in kwargs.items():
                print(arg,val)
            print('\n'*2)
    vprint('added model info to available_models.txt')
                

"""VERBOSITY"""
#TODO private
def tracefunc(frame, event, arg, indent=[0]):
          script=frame.f_code.co_filename
          fname=frame.f_code.co_name
          if config.PROJECT in script and (('>' or '<') not in fname):#naive way to ignore lambdas, etc
              if event == "call":
                  indent[0] += 2
                  print("-" * indent[0] + "> call function", fname)
              elif event == "return":
                  print("<" + "-" * indent[0], "exit function", fname)
                  indent[0] -= 2
              return tracefunc
if config.TRACEBACK:
    import sys
    sys.setprofile(tracefunc)

def readonly(filename):  os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)

def readwrite(filename): os.chmod(filename, S_IWUSR|S_IREAD) # This makes the file read/write for the owner

def savemodel(config,model,**kwargs):
    timestamp = str(dt.now())[:19]
    timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars
    NOW = re.sub(r'[\s]','_', timestamp)
    if not os.path.exists(config.MODELPATH):
        os.mkdir(config.MODELPATH)
    modelname=config.ALGORITHM+NOW
    modelfilename=modelname+'.joblib'
    configname=config.USEDCONFIGPATH+modelname+'.json'
    print('configname',configname)
    os.chdir(config.MODELPATH)
    print('dump',config.MODELPATH+modelfilename)
    dump(model, config.MODELPATH+modelfilename)
    os.chdir(config.USEDCONFIGPATH)
    attrs=saveconfig(config,configname)
    util.info(modelfilename,conf=attrs,**kwargs)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def saveconfig(config,configname):
    settings=config.__dict__['_wrapped'].default_settings.__dict__
    attrs={k:v for k,v in settings.items() if k==k.upper()}#TODO POTENTIALLY INCLUDE MORE FIELDS
    for k,v in attrs.items():
        if not is_jsonable(v):
            attrs[k]=str(v)
    with open(configname,'w') as f:
        json.dump(attrs, f)
    return attrs
    
        
            