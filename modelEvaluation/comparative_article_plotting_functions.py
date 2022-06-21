#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:47:31 2022

@author: aolza
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster

chosen_config='configurations.cluster.logistic'
experiment='configurations.urgcms_excl_nbinj'
import importlib
importlib.invalidate_caches()
from python_settings import settings as config
logistic_settings=importlib.import_module(chosen_config,package='estratificacion')
if not config.configured:
    config.configure(logistic_settings) # configure() receives a python module
assert config.configured 
import configurations.utility as util
util.makeAllPaths()

from modelEvaluation.calibrate import calibrate

import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import average_precision_score,RocCurveDisplay,roc_curve, auc,precision_recall_curve, PrecisionRecallDisplay

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def model_labels(models):
    labels=[]
    #The abbreviations that should appear in tables and figures:
    dictionary={'logistic':'LR','randomForest':'RF',
                'neuralNetworkRandom':'MLP','hgb':'GBDT'}
    for m in models:
        alg=re.sub('_|[0-9]', '', m)
        labels.append(dictionary[alg])
    return labels

def ROC_PR_comparison(models, yr, logistic_model, mode='ROC', **kwargs):
    # load logistic predictions
    parent=calibrate(logistic_model, yr)
    display={}
    # load models predictions
    models.append(logistic_model)
    labels=model_labels(models)
    for m, label in zip(models, labels): 
        print(m, label)
        if m==logistic_model:
            model=calibrate(m, yr,experiment_name=config.EXPERIMENT,
                            )
        else:
            predpath=re.sub(config.EXPERIMENT,'hyperparameter_variability_'+config.EXPERIMENT,config.PREDPATH)
            model=calibrate(m, yr, experiment_name='hyperparameter_variability_'+config.EXPERIMENT,
                            filename=os.path.join(predpath,f'{m}_calibrated_{yr}.csv'))
        obs=np.where(model.OBS>=1,1,0)
        fpr, tpr, _ = roc_curve(obs, model.PREDCAL)
        prec, rec, _ = precision_recall_curve(obs, model.PREDCAL)
        roc_auc = auc(fpr, tpr)
        avg_prec = average_precision_score(obs, model.PREDCAL)
        if mode=='PR':
            display[label]=PrecisionRecallDisplay(prec, rec, 
                                     estimator_name=label,
                                     average_precision=avg_prec)
        elif mode=='ROC':
            display[label]= RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                      estimator_name=label)
    import matplotlib.pyplot as plt
    fig2, ax = plt.subplots(1,1,figsize=(10,10))
    for curve in display.values():
        curve.plot(ax)
    return(display)
def boxplots(df, violin, together, **kwargs):
    import seaborn as sns
    order=kwargs.get('order',None)
    hue=kwargs.get('hue',None)
    supplementary=kwargs.get('supplementary', False)
    df['AUC']=df['Score']
    labels={'randomForest':'RF',
                'neuralNetworkRandom':'MLP','hgb':'GBDT'}
    # path=kwargs.get('path',config.FIGUREPATH)
    # name=kwargs.get('name','')
    # X=kwargs.get('X',None)
    # y=kwargs.get('y',None)

    # df['Algorithm']=[re.sub('_|[0-9]', '', model) for model in df['Model'].values]
    parent_metrics=df.copy().loc[df.Algorithm=='logistic']
    df=df.loc[df.Algorithm!='logistic']
    df=df.replace({'Algorithm': labels},regex=True)
    
    # fig, ((ax1,ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(10,12), gridspec_kw={'width_ratios': [1,1,1,3]})
    
    # if together:
    #     fig=plt.figure(figsize=(8.5,5.8))
    #     nrow=2
    #     ncol=3
    #     ax1 = plt.subplot2grid((nrow,ncol),(0,0))
    #     ax2 = plt.subplot2grid((nrow,ncol),(0,1))
    #     ax3 = plt.subplot2grid((nrow,ncol),(1,0))
    #     ax4 = plt.subplot2grid((nrow,ncol),(1,1))
    #     ax5 = plt.subplot2grid((nrow,ncol),(0,2), rowspan=2, colspan=1)
    #     for metric, ax in zip(['AUC', 'AP', 'Recall_20000', 'PPV_20000'],[ax1,ax2,ax3, ax4]):
    #         if not violin:
    #             df.boxplot(column=metric, by='Algorithm', ax=ax)
    #         if violin:
    #             sns.violinplot(ax=ax,x="Algorithm", y=metric, data=df,hue=hue)
    
    #         print(parent_metrics[metric].values[0])
    #         ax.axhline(y = parent_metrics[metric].values[0], linestyle = '-', label='Logistic', color='r')
    # else:
    for metric in ['AUC', 'AP', 'Recall_20000', 'PPV_20000']:
        fig, ax=plt.subplots()
        if not violin:
            df.boxplot(column=metric, by='Algorithm', ax=ax)
        if violin:
            sns.violinplot(ax=ax,x="Algorithm", y=metric,hue=hue, data=df)
            
        print(parent_metrics[metric].values[0])
        ls='--' if supplementary else '-'
        c=['blue', 'orange'] if supplementary else 'red'
        if supplementary:
            ax.axhline(y = parent_metrics.loc[parent_metrics.Year==2017][metric].values[0], 
                       linestyle = ls, label='LR 2017', color=c[0])
            ax.axhline(y = parent_metrics.loc[parent_metrics.Year==2018][metric].values[0], 
                       linestyle = ls, label='LR 2018', color=c[1])
        else:
            ax.axhline(y = parent_metrics[metric].values[0], linestyle = ls, label='Logistic', color=c)
        plt.legend()
    df['Before/After']='After'
    dff=df.copy()
    dff['Before/After']='Before'
    dff.Brier=dff['Brier Before']
    df2=pd.concat([dff,df])
    # df2['Before/After']='After'
    # df2.loc[original_index, 'Before/After']='Before'
    # df2.loc[original_index, 'Brier']=df2.loc[original_index, 'Brier Before']
    if together:
        df2.boxplot(column='Brier', by=['Before/After','Algorithm'], ax=ax5)
        ax5.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r')
        plt.legend()
        # plt.savefig(os.path.join(path,f'hyperparameter_variability_{'Brier'}.png'))
    else:
        df['Brier Change']=(-df['Brier Before']+df['Brier'])
        fig, ax5=plt.subplots()
    if violin:
        
        # ax5.axhline(y =-parent_metrics['Brier Before'].values[0]+parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r')
        ax5.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r')
        sns.violinplot(ax=ax5,x="Algorithm", y='Brier', hue='Before/After', data=df)
        fig, ax6=plt.subplots()
        ax6.axhline(y =parent_metrics['Brier Before'].values[0], linestyle = '-', label='LR Before', color='grey')
        ax6.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='LR After', color='r')
        sns.violinplot(ax=ax6,x="Algorithm", y='Brier', hue='Before/After', data=dff)
        # plt.yscale('logit')
    else:
        ax5.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r', alpha=0.9)
        df2.boxplot(column='Brier', by=['Before/After','Algorithm'], ax=ax5)
        
        plt.legend()
    plt.suptitle('')
    plt.tight_layout()
    plt.show()
    
def brier_boxplot_zoom(df, violin=True):
    import seaborn as sns
    labels={'randomForest':'RF',
                'neuralNetworkRandom':'MLP','hgb':'GBDT'}

    # df['Algorithm']=[re.sub('_|[0-9]', '', model) for model in df['Model'].values]
    parent_metrics=df.copy().loc[df.Algorithm=='logistic']
    df=df.loc[df.Algorithm!='logistic']
    df=df.replace({'Algorithm': labels})
    
   
    fig, ax=plt.subplots()
       
      
    df['Before/After']='After'
    dff=df.copy()
    dff['Before/After']='Before'
    dff.Brier=dff['Brier Before']
    df2=pd.concat([dff,df])
    # df2['Before/After']='After'
    # df2.loc[original_index, 'Before/After']='Before'
    # df2.loc[original_index, 'Brier']=df2.loc[original_index, 'Brier Before']
    if violin:
        data=df2.copy()
        data['Algorithms']=data['Algorithm']+' '+data['Before/After']
        sns.violinplot(ax=ax,x="Algorithms", y='Brier',
                       data=data,scale='count',hue='Algorithm',
                       order=['RF After', 'GBDT After', 'MLP After',
                              'MLP Before', 'RF Before', 'GBDT Before'])
    else:
        df2.boxplot(column='Brier', by=['Before/After','Algorithm'],
                    positions=[0,2,1,4,3,5],
                    ax=ax)
    ax.set_title('Variability of Brier Scores')

    # ax.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='Logistic', color='r')
    # ax.axhline(y =parent_metrics['Brier Before'].values[0], linestyle = '-', label='Logistic', color='purple')
    
    x1 = 3.5
    x2 = 5.3
    
    # select y-range for zoomed region
    y1 = 0.179
    y2 = 0.22
    
    # Make the zoom-in plot:
     # axins.plot(ts)
    if violin:
        axins = inset_axes(ax, 4.5,5 , loc='upper left', bbox_to_anchor=(0,0), borderpad=3,
                           bbox_transform=ax.figure.transFigure,
                           ) # zoomed_inset_axes(ax, 2, loc=1) # zoom = 2
        plt.setp(axins.spines.values(), color='green')
        plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='green')
        data=df2.copy()
        data['Algorithms']=data['Algorithm']+' '+data['Before/After']
        sns.violinplot(ax=axins,x="Algorithms", y='Brier',scale='count',
                       data=data,hue='Algorithm',
                       order=['RF After', 'GBDT After', 'MLP After',
                              'MLP Before', 'RF Before', 'GBDT Before'])
        # x1, x2=
        # y1, y2=0.175, 0.225
        axins.set_xlim(x1, x2)
        axins.set_ylim(0.175, 0.225)
        axins.set_title('Zoom', y=1.0, pad=-14)
        plt.xticks(visible=True)
        plt.yticks(visible=True)
        mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="green", ls='--')
        plt.draw()
        
    else:
        axins = inset_axes(ax, 2.5,5 , loc='upper left', bbox_to_anchor=(0,0), 
                           borderpad=3,bbox_transform=ax.figure.transFigure,
                           ec='green', ls='--') # zoomed_inset_axes(ax, 2, loc=1) # zoom = 2
        plt.setp(axins.spines.values(), color='green')
        plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='green')
        df2.boxplot(column='Brier', by=['Before/After','Algorithm'],
                    positions=[0,2,1,4,3,5], ax=axins)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_title('Zoom', y=1.0, pad=-14)
        plt.xticks(visible=True)
        plt.yticks(visible=True)
        mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="green", ls='--')
        plt.draw()
        
    # Make the zoom-in plot:
    x1 = -0.50
    x2 = 3.5
    
    # select y-range for zoomed region
    y1 = 0.047
    y2 = 0.049
    fig.subplots_adjust(left=1.2,right=1.3 ,bottom=1.4, top=1.5)

    axins2 = inset_axes(ax, 3.8,4.1 , loc='lower right', bbox_to_anchor=(0,0), borderpad=3,
                        bbox_transform=ax.figure.transFigure,
                      ) 
    # axins.plot(ts)
    if violin:
        data=df2.copy()
        data['Algorithms']=data['Algorithm']+' '+data['Before/After']
        sns.violinplot(ax=axins2,x="Algorithms", y='Brier',scale='count',
                       data=data,hue='Algorithm',
                       order=['RF After', 'GBDT After', 'MLP After',
                              'MLP Before', 'RF Before', 'GBDT Before'])
        
    else:
        df2.boxplot(column='Brier', by=['Before/After','Algorithm'],
                    positions=[0,2,1,4,3,5], ax=axins2)
    axins2.axhline(y =parent_metrics['Brier'].values[0], linestyle = '-', label='LR After', color='r')
    axins2.axhline(y =parent_metrics['Brier Before'].values[0], linestyle = '-', label='LR Before', color='purple')
    plt.setp(axins2.spines.values(), color='green')
    plt.setp([axins2.get_xticklines(), axins2.get_yticklines()], color='green')
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.set_title('Zoom' , y=1.0, pad=-14)
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    mark_inset(ax, axins2, loc1=1, loc2=1, fc="none", ec='green', ls='--')
    plt.draw()
    plt.legend()
   
    # plt.legend()
    plt.suptitle('')
    plt.tight_layout()
    plt.show()