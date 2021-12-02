#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" UTILITY FUNCTIONS THAT HAVE TO DO WITH SETTINGS 
    should be imported as: import utility as util
"""

import os
from datetime import datetime 
from contextlib import redirect_stdout
from python_settings import settings as config

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
    now = datetime.now() # current date and time
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
    
 
