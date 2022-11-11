#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:32:34 2022

@author: aolza
Source: https://github.com/konosp/propensity-score-matching/blob/main/propensity_score_matching_v2.ipynb
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import pandas as pd
from dataManipulation.dataPreparation import getData
import numpy as np
import os
from time import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%%
filename=os.path.join(config.PREDPATH,'pairs.csv')
if not Path(filename).is_file():
    Xx=X#.sample(1000000)
    z=Xx['FEMALE']
    df_data =Xx
    Xx=Xx.drop(['FEMALE', 'PATIENT_ID'], axis=1)
    print('Sample size ',len(X), 'positive: ',sum(z))
    assert not 'AGE_85GT' in X.columns
    
  
    logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)
    

    def logit(x, eps=1e-16):
        if x==0:
            return -1e6
        if x==1:
            return 1e6
        logit_=np.log(x)-np.log(1-x)
        return logit_
    from sklearn.neighbors import NearestNeighbors
    t0=time()
    fit=logistic.fit(Xx, z)
    print('fitting time: ',time()-t0)
    
    propensity_scores=fit.predict_proba(Xx)[:,1]
    propensity_logit = np.array([logit(xi) for xi in propensity_scores])
    
    
    df_data.loc[:,'propensity_score'] = propensity_scores
    df_data.loc[:,'propensity_score_logit'] = propensity_logit
    df_data.loc[:,'outcome'] = y.iloc[Xx.index][config.COLUMNS].values
    
    
    
    """ MATCHING """
    
    # common_support = (propensity_logit > -20) & (propensity_logit < 30)
    caliper = np.std(propensity_logit) * 0.25
    
    print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))#Caliper (radius) is: 0.0579
    
    
    knn = NearestNeighbors(n_neighbors=10 , p = 2, radius=caliper,n_jobs=-1)
    knn.fit(propensity_logit.reshape(-1, 1))
    
    
    df_data.index=range(len(df_data))
    distances , indexes = knn.kneighbors(
        df_data.propensity_score_logit.to_numpy().reshape(-1, 1), \
        n_neighbors=10)
    
    print('For item 0, the 4 closest distances are (first item is self):')
    for ds in distances[0,0:4]:
        print('Element distance: {:4f}'.format(ds))
    print('...')
    
    def perfom_matching_v2(row, indexes, df_data): #MATCHES WITH REPLACEMENT!! 
        current_index = int(row['index']) # Obtain value from index-named column, not the actual DF index.
        prop_score_logit = row['propensity_score_logit']
        for idx in indexes[current_index,:]:
            if (current_index != idx) and (row.FEMALE == 1) and (df_data.loc[idx].FEMALE == 0):
                return int(idx)
   
   
    df_data['matched_element'] = df_data.reset_index().apply(perfom_matching_v2, axis = 1, args = (indexes, df_data))
    
    females=df_data.loc[~df_data.matched_element.isna()]#.drop_duplicates('matched_element')
    males=df_data.loc[df_data.index.isin(df_data.matched_element)]

    pairs=pd.concat([males, females])

    pairs.to_csv(filename, index=False)
    print('Saved ',os.path.join(config.DATAPATH,'pairs.csv'))
else:
    pairs=pd.read_csv(filename)
#%%
plot=True
if plot:
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
    # Density distribution of propensity score (logic) broken down by treatment status
    fig, ax = plt.subplots(1,2)
    fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
    sns.kdeplot(x = propensity_scores, hue = z , ax = ax[0])
    ax[0].set_title('Propensity Score')
    sns.kdeplot(x = propensity_logit, hue = z , ax = ax[1])
    ax[1].axvline(-0.4, ls='--')
    ax[1].set_title('Logit of Propensity Score')
    plt.show()
    plt.savefig(os.path.join(config.FIGUREPATH,'propensity_densities_before.png'))
    
    
    sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
    # Density distribution of propensity score (logic) broken down by treatment status
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
    sns.kdeplot(x = pairs.propensity_score, hue = pairs.FEMALE , ax = ax)
    ax.set_title('Propensity Score')
    plt.show()
    # plt.savefig(os.path.join(config.FIGUREPATH,'after.png'))

    df_data['binary_outcome']=(df_data.outcome>0)
    pairs['binary_outcome']=(pairs.outcome>0)
    fig, ax = plt.subplots(2,1)
    fig.suptitle('Comparison of {} split by outcome and treatment status'.format('propensity_score_logit'))
    sns.stripplot(data = df_data, y = 'binary_outcome', alpha=0.5, x = 'propensity_score_logit', hue = 'FEMALE', orient = 'h', ax = ax[0]).set(title = 'Before matching', xlim=(-6, 4))
    sns.stripplot(data = pairs, y = 'binary_outcome', alpha=0.5, x = 'propensity_score_logit', hue = 'FEMALE', ax = ax[1] , orient = 'h').set(title = 'After matching', xlim=(-6, 4))
    plt.subplots_adjust(hspace = 0.3)
    plt.show()
    

    sns.set(rc={'figure.figsize':(10,30)}, font_scale=1.0)
    def cohenD (tmp, metricName):
        treated_metric = tmp[tmp.FEMALE == 1][metricName]
        untreated_metric = tmp[tmp.FEMALE == 0][metricName]
        
        d = ( treated_metric.mean() - untreated_metric.mean() ) / np.sqrt(((treated_metric.count()-1)*treated_metric.std()**2 + (untreated_metric.count()-1)*untreated_metric.std()**2) / (treated_metric.count() + untreated_metric.count()-2))
        return d
    data = []
    cols = np.random.choice(X.columns, size=100, replace=False)
    # cols = ['Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S']
    for cl in cols:
        data.append([cl,'before', cohenD(df_data,cl)])
        data.append([cl,'after', cohenD(pairs,cl)])
    
    res = pd.DataFrame(data, columns=['variable','matching','effect_size'])
    
    sn_plot = sns.barplot(data = res, y = 'variable', x = 'effect_size', hue = 'matching', orient='h')
    sn_plot.set(title='Standardised Mean differences accross covariates before and after matching')
    # sn_plot.figure.savefig("standardised_mean_differences.png")

#%%
overview = pairs[['outcome','FEMALE']].groupby(by = ['FEMALE']).aggregate([np.mean, np.var, np.std, 'count'])
print(overview.to_markdown())

treated_outcome = overview['outcome']['mean'][1]
treated_counterfactual_outcome = overview['outcome']['mean'][0]
att = treated_outcome - treated_counterfactual_outcome
print('The Average Treatment Effect (ATT): {:.4f}'.format(att))
#Vemos que, en una muestra de mujeres y hombres con las mismas 
#características clínicas, las mujeres ingresan menos (ATT<0)
#%%
ypairs=pairs.outcome
x=pairs[X.columns]
try:
    x.drop(['PATIENT_ID'],axis=1,inplace=True)
except:
    pass

ypairs=np.where(ypairs>=1,1,0)
ypairs=ypairs.ravel()
#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)
t0=time()
fit=logistic.fit(x, ypairs)
print('fitting time: ',time()-t0)
#%%
util.savemodel(config, fit,name='logistic_psm')
#%%
yMale=np.where(pairs.loc[pairs.FEMALE==0].outcome>=1,1,0)
yFemale=np.where(pairs.loc[pairs.FEMALE==1].outcome>=1,1,0)
from sklearn.metrics import roc_auc_score, average_precision_score
predsMale=fit.predict_proba(x.loc[x.FEMALE==0])
print('AUC Male= ',roc_auc_score(yMale,predsMale[:,1]))
print('AP Male= ',average_precision_score(yMale,predsMale[:,1]))

predsFemale=fit.predict_proba(x.loc[x.FEMALE==1])
print('AUC Female= ',roc_auc_score(yFemale,predsFemale[:,1]))
print('AP Female= ',average_precision_score(yFemale,predsFemale[:,1]))