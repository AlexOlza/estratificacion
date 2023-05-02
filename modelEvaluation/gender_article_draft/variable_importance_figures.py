#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:04:34 2023

@author: aolza
"""
import sys
import os
import configurations.utility as util
from python_settings import settings as config
chosen_config='configurations.cluster.'+sys.argv[1]
experiment='configurations.'+sys.argv[2]
import importlib
importlib.invalidate_caches()

logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

import joblib
from dataManipulation.dataPreparation import getData
import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
figurepath='/home/aolza/Desktop/estratificacion/figures/gender_article_draft/'
CCS=eval(input('CCS? True/False: '))
ccs='CCS' if CCS else 'ACG'
#%%
X,y=getData(2017, columns=['urgcms'])
X16,y17=getData(2016, columns=['urgcms'])
#%%
if CCS:
    logistic_modelpath=config.ROOTPATH+'models/urgcmsCCS_parsimonious/'
    linear_modelpath=config.ROOTPATH+'models/costCCS_parsimonious/'
    
    logistic_modelname='logistic20230324_111354'
    linear_modelname='linear20230324_130625'
    
    logistic_predpath=re.sub('models','predictions',logistic_modelpath)
    linear_predpath=re.sub('models','predictions',linear_modelpath)
else: #ACG
    logistic_modelpath=config.ROOTPATH+'models/urgcms_excl_nbinj/'
    linear_modelpath=config.ROOTPATH+'models/cost_ACG/'
    
    logistic_modelname='logistic20220705_155354'
    linear_modelname='linear20221018_103900'
    
    logistic_predpath=re.sub('models','predictions',logistic_modelpath)
    linear_predpath=re.sub('models','predictions',linear_modelpath)
logistic_global_model=joblib.load(logistic_modelpath+f'{logistic_modelname}.joblib')
linear_global_model=joblib.load(linear_modelpath+f'{linear_modelname}.joblib')
logistic_women_model=joblib.load(logistic_modelpath+f'logisticMujeres.joblib')
linear_women_model=joblib.load(linear_modelpath+f'linearMujeres.joblib')
logistic_men_model=joblib.load(logistic_modelpath+f'logisticHombres.joblib')
linear_men_model=joblib.load(linear_modelpath+f'linearHombres.joblib')
logistic_sameprev_model=joblib.load(logistic_modelpath+'logistic_gender_balanced.joblib')

descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')


log_global_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_global_model.feature_names_in_, logistic_global_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()
log_women_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_women_model.feature_names_in_, logistic_women_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()
log_men_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(logistic_men_model.feature_names_in_, logistic_men_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()

lin_global_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(linear_global_model.feature_names_in_, linear_global_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()
lin_women_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(linear_women_model.feature_names_in_, linear_women_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()
lin_men_coefs=pd.merge(pd.DataFrame.from_dict({name:[val] for name,val in zip(linear_men_model.feature_names_in_, linear_men_model.coef_[0])},
                                        orient='index',columns=['beta']),descriptions,left_index=True,right_on='CATEGORIES',how='left').reset_index()

print('logistic global female coef:',log_global_coefs.loc[log_global_coefs.CATEGORIES=='FEMALE'].beta.values )
print('linear global female coef:', lin_global_coefs.loc[lin_global_coefs.CATEGORIES=='FEMALE'].beta.values  )

#%%

N_women17=pd.Series([X.loc[X[c]==1].FEMALE.sum() for c in X],index=X.columns)

N_men17=pd.Series([len(X.loc[X[c]==1])-N_women17[c] for c in X],index=X.columns)

N_women=pd.Series([X16.loc[X16[c]==1].FEMALE.sum() for c in X16],index=X16.columns)

N_men=pd.Series([len(X16.loc[X16[c]==1])-N_women[c] for c in X16],index=X16.columns)


def tidy_up(df):
    df=df.reset_index()[['beta','LABELS','CATEGORIES']]
    df.loc[:,'LABELS']=np.where(df.LABELS.isna(),df.CATEGORIES,df.LABELS)
    df.loc[:,'Men']=[N_men[c] for c in df.CATEGORIES.values]
    df.loc[:,'Men17']=[N_men17[c] for c in df.CATEGORIES.values]
    df.loc[:,'Women']=[N_women[c] for c in df.CATEGORIES.values]
    df.loc[:,'Perc. Women']=100*df.Women/(df.Women+df.Men)
    df=df[['beta','LABELS','CATEGORIES','Men','Men17','Women','Perc. Women']]
    congestive_beta=df.loc[df.LABELS.str.startswith('PHARMA_Congestive_heart_failure_')].beta.sum()
    df.loc[len(df)] =[congestive_beta, 'PHARMA_Congestive_heart_failure','PHARMA_Congestive_heart_failure',
                      N_men['PHARMA_Congestive_heart_failure_block_1'],
                      N_men17['PHARMA_Congestive_heart_failure_block_1'],
                      N_women['PHARMA_Congestive_heart_failure_block_1'],df.loc[df.LABELS=='PHARMA_Congestive_heart_failure_block_1','Perc. Women'].values[0]]
    df.drop(df.loc[df.LABELS.str.startswith('PHARMA_Congestive_heart_failure_block_')].index,axis=0,inplace=True)
    
    return df

log_global_coefs,log_women_coefs,log_men_coefs,lin_global_coefs,lin_women_coefs,lin_men_coefs = (df.pipe(tidy_up) for df in [log_global_coefs,log_women_coefs,log_men_coefs,lin_global_coefs,lin_women_coefs,lin_men_coefs])

#%%
""" HOSPITALIZATION """
import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
def plot(coefs_,filename):
    coefs=coefs_.copy()
    sns.set( font_scale=2.0)
    sns.set_style("white")
    fig2,ax2=plt.subplots(1,1,figsize=(18,8))
    
    _negative=coefs.loc[coefs.beta<0].sort_values(by='beta',ascending=True)
    dropped=pd.concat([coefs[coefs.beta>1e10],_negative[_negative.beta<-1e10]])
    coefs.drop(dropped.index,inplace=True)
    _plot=coefs.nlargest(15,'beta').reset_index()
    # sns.barplot(data=_plot,x='beta',y='LABELS',hue='Perc. Women',orient='h')
    norm = plt.Normalize(0,100)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    rgba = sm.to_rgba( _plot['Perc. Women'].values)
    sm.set_array([])
    sns.barplot(data=_plot,x='beta',y='LABELS',palette=rgba,ax=ax2,orient='h')
    wrap_labels(ax2, width=50)
    ax2.set_ylabel('')
    ax2.figure.colorbar(sm)
    for p,t in zip(ax2.patches,_plot['Perc. Women'].values):
        ax2.annotate("%d" % t, xy=(p.get_width(), p.get_y()+p.get_height()/2.),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
    fig2.tight_layout()
    plt.savefig(figurepath+filename+'.jpeg',dpi=300)
    if len(dropped)>0:
        print(dropped[['LABELS','beta','Perc. Women','Men', 'Men17','Women']].to_markdown(index=False))
    

def plot_negative(coefs_,filename):
    coefs=coefs_.copy()
    sns.set( font_scale=2.0)
    sns.set_style("white")
    fig2,ax2=plt.subplots(1,1,figsize=(20,10))
    
    _negative=coefs.loc[coefs.beta<0].sort_values(by='beta',ascending=True)
    dropped=_negative[_negative.beta<-1e10]
    _negative.drop(dropped.index,inplace=True)
    _plot=_negative.nsmallest(15,'beta').reset_index()
    # sns.barplot(data=_plot,x='beta',y='LABELS',hue='Perc. Women',orient='h')
    norm = plt.Normalize(0,100)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    rgba = sm.to_rgba( _plot['Perc. Women'].values)
    sm.set_array([])
    sns.barplot(data=_plot,x='beta',y='LABELS',palette=rgba,ax=ax2,orient='h')
    wrap_labels(ax2, width=50)
    ax2.set_ylabel('')
    ax2.figure.colorbar(sm)
    for p,t in zip(ax2.patches,_plot['Perc. Women'].values):
        ax2.annotate("%d" % t, xy=(p.get_width(), p.get_y()+p.get_height()/2.),
            xytext=(-35, 0), textcoords='offset points', ha="left", va="center")
    fig2.tight_layout()
    plt.savefig(figurepath+filename+'.jpeg',dpi=300)
    if len(dropped)>0:
        print(dropped[['LABELS','beta','Perc. Women','Men', 'Men17','Women']].to_markdown(index=False))
    
#%%
plot(log_global_coefs,'variable_importance_log_global')
plot(lin_global_coefs,'variable_importance_lin_global')
plot(log_women_coefs,'variable_importance_log_women')
plot(lin_women_coefs,'variable_importance_lin_women')
plot(log_men_coefs,'variable_importance_log_men')
plot(lin_men_coefs,'variable_importance_lin_men') 
""" plot(lin_men_coefs,'variable_importance_lin_men')  dropped
                                          LABELS  Perc. Women  Men  Women
111  Fetal distress and abnormal forces of labor    99.911426    2   2256
184                              Cancer of ovary   100.000000    0   1501
"""
#%%
plot_negative(log_global_coefs,'variable_importance_negative_log_global')
plot_negative(lin_global_coefs,'variable_importance_negative_lin_global')
plot_negative(log_women_coefs,'variable_importance_negative_log_women')
plot_negative(lin_women_coefs,'variable_importance_negative_lin_women')
plot_negative(log_men_coefs,'variable_importance_negative_log_men')
plot_negative(lin_men_coefs,'variable_importance_negative_lin_men') 