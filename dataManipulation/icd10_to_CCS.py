#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:27:22 2022

@author: aolza
"""
import re
import pandas as pd
import time
import numpy as np
from python_settings import settings as config

import configurations.utility as util
configuration=util.configure()
from dataManipulation.dataPreparation import getData


IDCFILES={2016:'dx_in_2016.txt',
          2017:'dx_in_2017.csv'}
ICDTOCCSFILE='ccs_dx_icd10cm_2017.csv'
def CCSData(yr,
            **kwargs):
    from dataManipulation.dataPreparation import getData 
    ccs=pd.read_csv(config.INDISPENSABLEDATAPATH+'ccs/'+ICDTOCCSFILE, dtype=str)
    
    for c in ccs:
        print(f'{c} has {len(ccs[c].unique())} unique values')
    return 0,0


