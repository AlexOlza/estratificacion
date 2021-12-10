def try_import(statements):
    for s in statements: 
        try:
            exec(s)
        except: 
            print('error: ',s)
statements=['from pathlib import Path',
'from warnings import warn',
'import pandas as pd',
'import numpy as np',
'import time',
'import os',
'import pyreadr',
'import re',
'import csv',
'import importlib',
'from datetime import timedelta,datetime',
'import sys   ',
'from contextlib import redirect_stdout',
'from stat import S_IREAD, S_IRGRP, S_IROTH',
'from joblib import dump',
'from datetime import datetime as dt',
'from sklearn.ensemble import RandomForestClassifier',
'from sklearn.model_selection import train_test_split',
'from sklearn.model_selection import RandomizedSearchCV',
'from stat import S_IWUSR # Need to add this import to the ones above',
'import shutil',
'import re',
'import json',
'from contextlib import redirect_stdout',
'from python_settings import SetupSettings',
'from python_settings import settings as config',
'from configurations.default import *',
'from configurations.cluster.default import *',
'from configurations.cluster import configRandomForest as randomForest_settings',
'config.configure(randomForest_settings)',
'import configurations.utility as util',
'from dataManipulation.generarTablasIngresos import createYearlyDataFrames, loadIng, assertMissingCols',
'from dataManipulation.generarTablasVariables import prepare,resourceUsageDataFrames,load',
'from configurations import utility as util',
'from dataManipulation.dataPreparation import getData',
'from configurations.utility import makeAllPaths ',
]
try_import(statements)
