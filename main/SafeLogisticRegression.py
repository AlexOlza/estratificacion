#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:41:00 2021

@author: aolza
"""
#%%
from sklearn import linear_model
class SafeLogisticRegression(linear_model.LogisticRegression):
    def __init__(self,penalty,max_iter,verbose):
        super().__init__(penalty=penalty,max_iter=max_iter,verbose=verbose) 
    def set_columns(self,columns):
        self.columns=columns
    def safeFit(self, X, y):#FIXME KERNEL BREAKS!!! dataconversionwarning
        self.columns = X.columns
        print(self.columns)
        return self.fit( X, y)
    def predict_proba(self, X):
        new_columns = list(X.columns)
        old_columns = list(self.columns)
        if new_columns != old_columns:
            if len(new_columns) == len(old_columns):
                try:
                    X = X[old_columns]
                    print( "The order of columns has changed. Fixed.")
                except:
                    raise ValueError('The columns has changed. Please check.')
            else:
                raise ValueError('The number of columns has changed.')
        return linear_model.LogisticRegression.predict_proba(self, X)