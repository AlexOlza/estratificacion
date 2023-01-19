#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

@author: aolza
source:
    https://fakhredin.medium.com/forward-selection-to-find-predictive-variables-with-python-code-3c33f0db2393
"""
from sklearn.linear_model import LinearRegression

def sms_forward_regression(df, y, candidates = ['AGE','GMA','FEMALE']): 
    import statsmodels.formula.api as sm
    ar2 = dict()
    last_max = -1
    
    while(True):
        for x in df.drop([y] + candidates, axis=1).columns:
            if len(candidates) == 0:
                features = x
            else:
                features = x + ' + '
                features += ' + '.join(candidates)
    
            model = sm.ols(y + ' ~ ' + features, df).fit()
            ar2[x] = model.rsquared_adj
    
        max_ar2 =  max(ar2.values())
        max_ar2_key = max(ar2, key=ar2.get)
    
        if max_ar2 > last_max:
            candidates.append(max_ar2_key)
            last_max = max_ar2
    
            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Adjusted R2: ' + str(max_ar2))
            print('===============')
        else:
            print(model.summary())
            break
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
#%%
def sklearn_forward_regression(df, y, candidates = ['AGE','GMA','FEMALE'], tol=1e-4):    
    ar2 = dict()
    last_max = -1
    
    while(True):
        # iteration i+1
        # for each column that is not already in the model
        for x in df.drop([y] + candidates, axis=1).columns:
            if len(candidates) == 0:
                features = x
            else:
                features = [x]+candidates
            # we try building a model with such column and the preexisting ones
            # we now have n_{i+1} columns
            model = LinearRegression(n_jobs=-1).fit(df[features],df[y])
            #we compute adjusted r squared
            ar2[x] = 1 - (1-model.score(df[features],df[y]))*(len(df[y])-1)/(len(df[y])-df[features].shape[1]-1)
        # after trying all potential next columns
       
        max_ar2 =  max(ar2.values())
        max_ar2_key = max(ar2, key=ar2.get)
        # we check whether any of them has increased performance
        if max_ar2 > last_max + tol:
            candidates.append(max_ar2_key) #if so, we add the best one
            last_max = max_ar2
    
            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Adjusted R2: ' + str(max_ar2))
            print('===============')
        else:
            break
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
    return model
#%%
def sklearn_stepwise_regression(df, y, minimal = [], tol=1e-4):    
    ar2 = dict()
    last_max = -1
    candidates = list( minimal.copy())
    if len(df.drop(list([y])+list(minimal), axis=1).columns)==0:
        model = LinearRegression(n_jobs=-1).fit(df[minimal],df[y])
        #we compute adjusted r squared
        ar2['minimal'] = 1 - (1-model.score(df[minimal],df[y]))*(len(df[y])-1)/(len(df[y])-df[minimal].shape[1]-1)
        print('Returning minimal model')
        print('Adjusted R2: ' + str(ar2['minimal']))
        return model
    while(True):
        # iteration i+1
        # for each column that is not already in the model
        for x in df.drop(list([y]) + list(candidates), axis=1).columns:
            if len(candidates) == 0:
                features = x
            else:
                features = [x]+list(candidates)
            # we try building a model with such column and the preexisting ones
            # we now have n_{i+1} columns
            model = LinearRegression(n_jobs=-1).fit(df[features],df[y])
            #we compute adjusted r squared
            ar2[x] = 1 - (1-model.score(df[features],df[y]))*(len(df[y])-1)/(len(df[y])-df[features].shape[1]-1)
        # after trying all potential next columns
       
        max_ar2 =  max(ar2.values())
        max_ar2_key = max(ar2, key=ar2.get)
        # we check whether any of them has increased performance
        if max_ar2 > last_max + tol:
            candidates.append(max_ar2_key) #if so, we add the best one
            last_max = max_ar2
    
            print('forward step: ' + str(len(candidates)))
            print(candidates)
            print('Adjusted R2: ' + str(max_ar2))
            
        else:
            break
        print('Testing variables to remove...')
        for x in [c for c in candidates if not c in minimal] :
            features = [c for c in candidates if c!=x]
            # we try building a model with such column and the preexisting ones
            # we now have n_{i+1} columns
            model = LinearRegression(n_jobs=-1).fit(df[features],df[y])
            #we compute adjusted r squared
            ar2[f'-{x}'] = 1 - (1-model.score(df[features],df[y]))*(len(df[y])-1)/(len(df[y])-df[features].shape[1]-1)
        # after trying all potential next columns
        print('===============')
        max_ar2 =  max(ar2.values())
        max_ar2_key = max(ar2, key=ar2.get)
        # we check whether any of them has increased performance
        if max_ar2 > last_max + tol:
            print(max_ar2_key)
            candidates.remove(max_ar2_key[1:]) #if so, we remove the best one
            last_max = max_ar2
    
            print('backward step: ' + str(len(candidates)) + max_ar2_key)
            print(candidates)
            print('Adjusted R2: ' + str(max_ar2))
            print('===============')
        else:
            continue
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
    return model