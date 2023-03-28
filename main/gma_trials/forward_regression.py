#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:53:52 2023

@author: aolza
source:
    https://fakhredin.medium.com/forward-selection-to-find-predictive-variables-with-python-code-3c33f0db2393
"""
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np
import time
import datetime

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
def sklearn_stepwise_regression_simple(df, y, minimal = [], tol=1e-4,algorithm='linear',**kwargs):    
    if algorithm=='logistic':
        return sklearn_stepwise_logistic_regression(df, y, minimal)
            
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

def sklearn_stepwise_logistic_regression(df, y, minimal= [], tol= 1000,**kwargs): 
    AIC = dict()
    last_min = np.inf
    candidates = list( minimal.copy())
    if len(df.drop(list([y])+list(minimal), axis=1).columns)==0:
        model = LogisticRegression(penalty='none',n_jobs=-1).fit(df[minimal],df[y])
        #we compute adjusted r squared
        #we compute AIC
        #the logloss is the negative log-likelihood
        y_pred=model.predict_proba(df[minimal])[:,1]
        AIC['minimal'] = 2*(len(minimal))+2*len(y_pred)*log_loss(df[y], y_pred)
        print('Returning minimal model')
        print('AIC: ' + str(AIC['minimal']))
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
            t0=time.time()
            model = LogisticRegression(penalty='none',n_jobs=-1).fit(df[features],df[y])
            #we compute AIC
            #the logloss is the negative log-likelihood
            y_pred=model.predict_proba(df[features])[:,1]
            
            AIC[x] = 2*(len(features))+2*len(y_pred)*log_loss(df[y], y_pred)
            t=time.time()
            print(f'Variable {x}, AIC {AIC[x]}, time {str(datetime.timedelta(seconds=t-t0))}')
            # after trying all potential next columns
        min_AIC =  min(AIC.values())
        min_AIC_key = min(AIC, key=AIC.get)
        # we check whether any of them has increased performance
        print('last min, new min:',last_min,min_AIC)
        if min_AIC < (last_min-tol ):
            candidates.append(min_AIC_key) #if so, we add the best one
            last_min = min_AIC
    
            print('forward step: ' + str(len(candidates)))
            print(candidates)
            print('AIC: ' + str(min_AIC))
            
        else:
            break
        # print('Testing variables to remove...')
        # for x in [c for c in candidates if not c in minimal] :
        #     features = [c for c in candidates if c!=x]
        #     # we try building a model with such column and the preexisting ones
        #     # we now have n_{i+1} columns
        #     model = LogisticRegression(penalty='none',n_jobs=-1).fit(df[features],df[y])
        #     #we compute AIC
        #     #the logloss is the negative log-likelihood
        #     y_pred=model.predict_proba(df[features])[:,1]
        #     AIC[f'-{x}'] = 2*(len(features)+1)+2*log_loss(df[y], y_pred)
        #     # after trying all potential next columns
        # print('===============')
        # min_AIC =  min(AIC.values())
        # min_AIC_key = min(AIC, key=AIC.get)
        # # we check whether any of them has increased performance
        # if min_AIC < (last_min - tol):
        #     print(min_AIC_key)
        #     candidates.remove(min_AIC_key[1:]) #if so, we remove the best one
        #     last_min = min_AIC
    
        #     print('backward step: ' + str(len(candidates)) + min_AIC_key)
        #     print(candidates)
        #     print('AIC: ' + str(min_AIC))
        #     print('===============')
        # else:
        #     continue
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
    return model

def sklearn_stepwise_regression(df, y, minimal = [], tol=1e-4, scoring='R2', test_set=False):    
    if test_set:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df.drop(y,axis=1), df[y], test_size=0.3, random_state=42)
        df_train=pd.concat([X_train, y_train],axis=1)
        df_test=pd.concat([X_test, y_test],axis=1)
    else:
        df_train=df
        df_test=df
    if scoring=='R2':
        best=max
        score_name='Adjusted R2: '
        last_best = -1
        def condition(best_score,last_best,tol): return best(best_score,last_best+tol) == best_score
    else:
        best=min
        score_name='RMSE: '
        last_best = 10e6
        def condition(best_score,last_best,tol): return best(best_score,last_best-tol) == best_score
    score = dict()
    
    candidates = list( minimal.copy())
    if len(df.drop(list([y])+list(minimal), axis=1).columns)==0:
        model = LinearRegression(n_jobs=-1).fit(df_train[minimal],df_train[y])
        if scoring=='R2':
            score_model=1 - (1-model.score(df_test[minimal],df_test[y]))*(len(df_test[y])-1)/(len(df_test[y])-df_test[minimal].shape[1]-1)
        else:
            score_model=mean_squared_error(df_test[y],model.predict(df_test[model.feature_names_in_]), squared=False)
        #we compute adjusted r squared
        score['minimal'] = score_model
        print('Returning minimal model')
        print(score_name + str(score['minimal']))
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
            model = LinearRegression(n_jobs=-1).fit(df_train[features],df_train[y])
            #we compute adjusted r squared
            if scoring=='R2':
                score_model=1 - (1-model.score(df_test[features],df_test[y]))*(len(df_test[y])-1)/(len(df_test[y])-df_test[features].shape[1]-1)
            else:
                score_model=mean_squared_error(df_test[y],model.predict(df_test[model.feature_names_in_]), squared=False)
            score[x] = score_model
        # after trying all potential next columns
       
        best_score =  best(score.values())
        best_score_key = best(score, key=score.get)
        # we check whether any of them has increased performance
        if condition(best_score,last_best,tol):
            candidates.append(best_score_key) #if so, we add the best one
            last_best = best_score
    
            print('forward step: ' + str(len(candidates)))
            print(candidates)
            print(score_name + str(best_score))
            
        else:
            break
        print('Testing variables to remove...')
        for x in [c for c in candidates if not c in minimal] :
            features = [c for c in candidates if c!=x]
            # we try building a model with such column and the preexisting ones
            # we now have n_{i+1} columns
            model = LinearRegression(n_jobs=-1).fit(df_train[features],df_train[y])
            #we compute adjusted r squared
            if scoring=='R2':
                score_model=1 - (1-model.score(df_test[features],df_test[y]))*(len(df_test[y])-1)/(len(df_test[y])-df_test[features].shape[1]-1)
            else:
                score_model=mean_squared_error(df_test[y],model.predict(df_test[model.feature_names_in_]), squared=False)
            score[f'-{x}']  = score_model
            
        # after trying all potential next columns
        print('===============')
        best_score =  best(score.values())
        best_score_key = best(score, key=score.get)
        # we check whether any of them has increased performance
        # if best_score > last_best + tol:
        if condition(best_score,last_best,tol):
            print(best_score_key)
            key_to_remove=best_score_key[1:] if best_score_key[0]=='-' else best_score_key
            candidates.remove(key_to_remove) #if so, we remove the best one
            last_best = best_score
    
            print('backward step: ' + str(len(candidates)) + best_score_key)
            print(candidates)
            print(score_name + str(best_score))
            print('===============')
        else:
            continue
    
    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))
    return model