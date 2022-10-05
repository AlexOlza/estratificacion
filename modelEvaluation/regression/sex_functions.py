#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import norm
import re
import os
import joblib as job
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')
import pandas as pd
#%%
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
try:
    usedconfigpath=os.environ['USEDCONFIG_PATH']
except:
    usedconfigpath=sys.argv[3]

configfile=input('Config file name')# linearMujeres.json, neuralRegressionMujeres.json...
config_used=os.path.join(usedconfigpath,f'{sys.argv[2]}/{configfile}')

from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configuration=util.configure(config_used)
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
from modelEvaluation.independent_sex_riskfactors import translateVariables
#%%
def beta_std_error(linModel, X, y, eps=1e-20):
    """ Source:
        A Modern Approach to Regression with R, Chapter 5; Sheather 2009
    """
    #   Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])             #shape n*(p+1)
    beta=list(linModel.intercept_)+list(linModel.coef_[0])
    residual_sum_squares=(y-X_design*beta)**2
    sigma_squared=residual_sum_squares/(len(X)-len(X.columns)-1) 
    XtX=X_design.T@X_design
    beta_var_cov=sigma_squared*np.linalg.pinv(XtX)
    D=np.diag(beta_var_cov).copy() #copy, because the original array is non-mutable
    beta_std_error_=np.sqrt(D)
    return beta_std_error_

def confidence_interval_odds_ratio(betas, stderr, confidence_level):
    """ Source:
        https://stats.stackexchange.com/questions/354098/calculating-confidence-intervals-for-a-logistic-regression
     Using the invariance property of the MLE allows us to exponentiate to get the conf.int.
    """
    low=betas-norm.interval(confidence_level)[1]*stderr
    high=betas+norm.interval(confidence_level)[1]*stderr
    return(low,high)
