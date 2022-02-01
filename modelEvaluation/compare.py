#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:13:19 2021

@author: aolza


    Input: Experiment name, prediction year
    In the models/experiment directory, detect...
        All algorithms present
        The latest model for each algorithm
    Prompt user for consent about the selected models
    Load models
    Predict (if necessary, i.e. if config.PREDPATH+'/{0}__{1}.csv'.format(model_name,yr) is not found )
    Save into big dataframe (not sure this is clever)
    Calibrate (coming soon)
    Compare

"""

import pandas as pd
from pathlib import Path

from python_settings import settings as config
import configurations.utility as util
util.configure('configurations.local.logistic')

year=int(input('YEAR YOU WANT TO PREDICT: '))
assert year in [2017,2018,2019], 'No data available!'
list((Path(config.MODELPATH).glob('**/*')))

available_models=[x.parts[-1] for x in Path(config.MODELPATH).glob('**/*') if x.is_file()]


        
