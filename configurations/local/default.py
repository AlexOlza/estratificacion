#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:29:36 2021

@author: aolza
"""
"""LOCATIONS"""
import os
project='estratificacion'
rootPath='/home/aolza/Desktop/estratificacion/'
DATAPATH='/home/aolza/Desktop/estratificacionDatos/'
indispensableDataPath=dataPath+'indispensable/'
outPath=rootPath+'predictions/'
modelPath=rootPath+'models/'
"""FILES"""
allHospitFile='ingresos2016_2018.csv'
ACGfiles={2016:'ing2016-2017Activos.csv',
          2017:'ing2017-2018Activos.csv',
          2018:'ing2018-2019Activos.csv'}
ACGIndPrivFiles={2016:'ing2016-2017ActivosIndPriv.csv',
                 2017:'ing2017-2018ActivosIndPriv.csv',
                 2018:'ing2018-2019ActivosIndPriv.csv'}
"""PREDICTORS: They will be different for each script."""
# predictorRegex=r'PATIENT_ID|AGE_[0-9]+$|ACG|EDC_|HOSDOM|FRAILTY|RXMG_|INGRED_14GT|INDICE_PRIVACION'
# predictors=True
# indicePrivacion=True
# columns=[]
# previousHosp=[]

"""VERBOSITY SETTINGS"""
verbose=False 
traceback=False #EXTREME VERBOSITY 
vprint = print if verbose else lambda *a, **k: None
def tracefunc(frame, event, arg, indent=[0]):
      script=frame.f_code.co_filename
      fname=frame.f_code.co_name
      if project in script and (('>' or '<') not in fname):#naive way to ignore lambdas, etc
          if event == "call":
              indent[0] += 2
              print("-" * indent[0] + "> call function", fname)
          elif event == "return":
              print("<" + "-" * indent[0], "exit function", fname)
              indent[0] -= 2
          return tracefunc
if traceback:
    import sys
    sys.setprofile(tracefunc)
    
from datetime import datetime    
""" INFO FILE """
def info(predPath,experiment,algorithm,**kwargs):
    
    now = datetime.now() # current date and time

    # def ifprint(message,arg): print(message,arg) if arg else lambda *a, **k: None
    from contextlib import redirect_stdout
    file=os.path.join(predPath,'info_{0}.txt'.format(algorithm))
    with open(file, 'a+') as f:
        with redirect_stdout(f):
            print('Experiment date: ',now.strftime("%m/%d/%Y, %H:%M:%S"))
            print('Algorithm: ',algorithm)
            for arg,val in kwargs.items():
                print(arg,val)
            # ifprint('Parameter search space:',randomGrid)
            # ifprint('Number of trials:',trials)
            # ifprint('Best params:',)
            
