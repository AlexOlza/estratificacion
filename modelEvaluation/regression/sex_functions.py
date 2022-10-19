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
from sklearn.linear_model import LinearRegression, RidgeCV
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


from python_settings import settings as config
import configurations.utility as util
if not config.configured: 
    configfile=input('Config file name')# linearMujeres.json, neuralRegressionMujeres.json...
    config_used=os.path.join(usedconfigpath,f'{sys.argv[2]}/{configfile}')
    configuration=util.configure(config_used)
from dataManipulation.dataPreparation import getData
from modelEvaluation.predict import generate_filename
from modelEvaluation.independent_sex_riskfactors import translateVariables
#%%
def beta_std_error(linModel, Xx, yy, eps=1e-20):
    """ Source:
        A Modern Approach to Regression with R, Chapter 5; Sheather 2009
    """
    if isinstance(linModel, RidgeCV):
        return ridge_std_error(linModel, Xx, yy, eps=1e-20)
    #   Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((Xx.shape[0], 1)), Xx])             #shape n*(p+1)
    beta=list(linModel.intercept_)+list(linModel.coef_[0])
    residual_sum_squares=sum(yy.to_numpy()-X_design * beta)**2
    sigma_squared=residual_sum_squares/(len(Xx)-len(Xx.columns)-1) 
    XtX=X_design.T @ X_design                                           #shape (p+1)(p+1)
    beta_var_cov=sigma_squared*np.linalg.pinv(XtX)
    D=np.diag(beta_var_cov).copy() #copy, because the original array is non-mutable
    D[np.abs(D) < eps] = 0 # Avoid negative values in the diagonal (a necessary evil)
    beta_std_error_=np.sqrt(D)
    return beta_std_error_

def ridge_std_error(linModel, Xx, yy, eps=1e-20):
    """ Source:
        https://arxiv.org/pdf/1509.09169.pdf
    """
    print('ridge_std_error')
    #   Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones((Xx.shape[0], 1)), Xx])             #shape n*(p+1)
    beta=list(linModel.intercept_)+list(linModel.coef_[0])
    residual_sum_squares=sum(yy.to_numpy()-X_design * beta)**2
    sigma_squared=residual_sum_squares/(len(Xx)-len(Xx.columns)-1) 
    
    XtX=X_design.T @ X_design                                           #shape (p+1)(p+1)
    penalty=linModel.alpha_
    W_inverse=np.linalg.inv(XtX+penalty*np.identity(len(XtX)))
    beta_var_cov=sigma_squared*W_inverse @ XtX @ W_inverse.T
    D=np.diag(beta_var_cov).copy() #copy, because the original array is non-mutable
    # D[np.abs(D) < eps] = 0 # Avoid negative values in the diagonal (a necessary evil)
    beta_std_error_=np.sqrt(D)
    return beta_std_error_

def confidence_interval_betas(betas, stderr, confidence_level):
    """ Source:
        https://stats.stackexchange.com/questions/354098/calculating-confidence-intervals-for-a-logistic-regression
     Using the invariance property of the MLE allows us to exponentiate to get the conf.int.
    """
    low=betas-norm.interval(confidence_level)[1]*stderr
    high=betas+norm.interval(confidence_level)[1]*stderr
    return(low,high)
