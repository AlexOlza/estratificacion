#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:32:34 2022

@author: aolza
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

from dataManipulation.dataPreparation import getData
import numpy as np
from sklearn.linear_model import LogisticRegression

#%%
np.random.seed(config.SEED)

X,y=getData(2016)
#%%
Xx=X#.sample(1000000)
z=Xx['FEMALE']
df_data =Xx
Xx=Xx.drop(['FEMALE', 'PATIENT_ID'], axis=1)
print('Sample size ',len(X), 'positive: ',sum(z))
assert not 'AGE_85GT' in X.columns

#%%
logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0)

from time import time
from scipy.special import logit
from sklearn.neighbors import NearestNeighbors
t0=time()
fit=logistic.fit(Xx, z)
print('fitting time: ',time()-t0)
#%%
propensity_scores=fit.predict_proba(Xx)[:,1]
propensity_logit = np.array([logit(xi) for xi in propensity_scores])


df_data.loc[:,'propensity_score'] = propensity_scores
df_data.loc[:,'propensity_score_logit'] = propensity_logit
df_data.loc[:,'outcome'] = y.iloc[Xx.index][config.COLUMNS].values
#%%

#%%
""" MATCHING """

common_support = (propensity_logit > -20) & (propensity_logit < 30)
caliper = np.std(propensity_logit) * 0.25

print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))#Caliper (radius) is: 0.0579

#%%
knn = NearestNeighbors(n_neighbors=10 , p = 2, radius=caliper,n_jobs=-1)
knn.fit(propensity_logit.reshape(-1, 1))

#%%
df_data.index=range(len(df_data))
distances , indexes = knn.kneighbors(
    df_data.propensity_score_logit.to_numpy().reshape(-1, 1), \
    n_neighbors=10)

print('For item 0, the 4 closest distances are (first item is self):')
for ds in distances[0,0:4]:
    print('Element distance: {:4f}'.format(ds))
print('...')

#%%
def perfom_matching_v2(row, indexes, df_data): #MATCHES WITH REPLACEMENT!! 
    current_index = int(row['index']) # Obtain value from index-named column, not the actual DF index.
    prop_score_logit = row['propensity_score_logit']
    for idx in indexes[current_index,:]:
        if (current_index != idx) and (row.FEMALE == 1) and (df_data.loc[idx].FEMALE == 0):
            return int(idx)
"""
knn = NearestNeighbors(n_neighbors=len(dfB)).fit(dfB)
distances, indices = knn.kneighbors(dfA)

matched = []
pairs = []
for indexA, candidatesB in enumerate(indices):
    personA = dfA.index[indexA]
    for indexB in candidatesB:
        if indexB not in matched:
            matched.append(indexB)
            personB = dfB.index[indexB]
            pairs.append([personA, personB])
            break

matches = pd.DataFrame(pairs, columns=['SetA', 'SetB'])

"""
# def vectorized_matching(indexes, df_data):
#     pass
#     noself=[idx[1:] for ind in indexes] #exclude self as match       
#     noduplicates= [idx.pop()]
# def perfom_matching_v2(row, indexes, df_data): #MATCHES WITH REPLACEMENT!! 
#     current_index = int(row['index']) # Obtain value from index-named column, not the actual DF index.
#     prop_score_logit = row['propensity_score_logit']
#     for idx in indexes[current_index,:]:
#         if (current_index != idx) and (row.FEMALE == 1) and (df_data.loc[idx].FEMALE == 0):
#             return int(idx)

df_data['matched_element'] = df_data.reset_index().apply(perfom_matching_v2, axis = 1, args = (indexes, df_data))

females=df_data.loc[~df_data.matched_element.isna()].drop_duplicates('matched_element')
males=df_data.loc[df_data.index.isin(df_data.matched_element)]
import pandas as pd
pairs=pd.concat([males, females])
#%%
import os
pairs.to_csv(os.path.join(config.DATAPATH,'pairs.csv'), index=False)
#%%
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
plt.savefig(os.path.join(config.FIGUREPATH,'before.png'))


sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
# Density distribution of propensity score (logic) broken down by treatment status
fig, ax = plt.subplots(1,1)
fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
sns.kdeplot(x = pairs.propensity_score, hue = pairs.FEMALE , ax = ax)
ax.set_title('Propensity Score')
plt.show()
plt.savefig(os.path.join(config.FIGUREPATH,'after.png'))
#%%
