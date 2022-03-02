#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" UTILITY FUNCTIONS THAT HAVE TO DO WITH SETTINGS AND MODEL PERSISTENCE
    should be imported as: from configurations import utility as util
"""
    
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
from joblib import dump
import re
import json
import os
from datetime import datetime as dt
from contextlib import redirect_stdout

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
from python_settings import settings as config

def configure(configname=None,**kwargs):
    if config.configured and (not configname):
        print('Already configured with configname ',config.CONFIGNAME)
        TRACEBACK=kwargs.get('TRACEBACK', config.TRACEBACK)
        if TRACEBACK:
            import sys
            sys.setprofile(tracefunc)
        makeAllPaths()
        return None
    if not configname:
        configname=input('Enter configuration json file path: ')
    if configname.endswith('.json'):
        with open(configname) as c:
            configuration=json.load(c)
        for k,v in configuration.items():
            if isinstance(v,dict):#for example ACGFILES
                try:
                     configuration[k]= {int(yr):filename for yr,filename in v.items()}
                except Exception as exc:
                    print (exc)
        conf=Struct(**configuration)
        conf.TRACEBACK=kwargs.get('TRACEBACK', conf.TRACEBACK)
        conf.VERBOSE=kwargs.get('VERBOSE',conf.VERBOSE)
        if not config.configured:
            config.configure(conf) # configure() receives a python module
        if conf.TRACEBACK:
            import sys
            sys.setprofile(tracefunc)
    else:
        import importlib
        importlib.invalidate_caches()
        conf=importlib.import_module(configname,package='estratificacion')
        configuration=None
    # else:
    #     print('Provide .json or .py configuration file!')
    if not config.configured:
        config.configure(conf) # configure() receives a python module
    assert config.configured
    makeAllPaths()
    return configuration

# configuration=configure()
# vprint = print if config.VERBOSE else lambda *a, **k: None
def vprint(*args):
    print(*args) if config.VERBOSE else lambda *a, **k: None
    
 
def makeAllPaths():
    settings=config.__dict__['_wrapped'].default_settings.__dict__
    paths={k:v for k,v in settings.items() if 'PATH' in k}
    for p in paths.values():
        if not os.path.exists(p):
            os.makedirs(p)
            vprint('created directory ',p)
    vprint('All paths exist now')
        
""" INFO FILE"""
def info(modelfilename,**kwargs):
    modelPath=kwargs.get('modelPath',config.MODELSPATH)
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
          

def readonly(filename):  os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)

def readwrite(filename): os.chmod(filename, S_IWUSR|S_IREAD) # This makes the file read/write for the owner

def savemodel(config,model,**kwargs):
    timestamp = str(dt.now())[:19]
    timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars
    NOW = re.sub(r'[\s]','_', timestamp)
    if not os.path.exists(config.MODELPATH):
        os.mkdir(config.MODELPATH)
    modelname=kwargs.get('name',config.ALGORITHM+NOW)
    modelfilename=modelname+'.joblib'
    configname=config.USEDCONFIGPATH+modelname+'.json'
    print('configname',configname)
    os.chdir(config.MODELPATH)
    print('dump',config.MODELPATH+modelfilename)
    dump(model, config.MODELPATH+modelfilename)
    os.chdir(config.USEDCONFIGPATH)
    attrs=saveconfig(config,configname)
    info(modelfilename,conf=attrs,**kwargs)

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
    
        
            