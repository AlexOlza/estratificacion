#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:01:28 2023

@author: aolza
"""
"""
We exclude 444162 patients, 65.18094749213125 % women
Sample size  1810090 , female percentage:  47.5460888685093
Initial AUC for propensity scores:  0.7095418287008433

a reference: https://sci-hub.se/10.1213/ANE.0000000000002787
Created on Tue Nov  8 10:32:34 2022

@author: aolza
Source: https://github.com/konosp/propensity-score-matching/blob/main/propensity_score_matching_v2.ipynb
"""
import sys
sys.path.append('/home/aolza/Desktop/estratificacion/')#necessary in cluster
# cost=eval(input('Enter True for COST or False for HOSPITALIZATION: '))
# exp= 'costCCS_vigo' if cost else 'urgcmsCCS_vigo'
# conf='linear' if cost else 'logistic'
# chosen_config='configurations.cluster.'+conf
# experiment='configurations.'+exp
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
import joblib as job
from dataManipulation.dataPreparation import getData
from modelEvaluation.compare import performance
import numpy as np
import os
from time import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
#%%
def logit(x, eps=1e-16):
    if x==0:
        return -1e6
    if x==1:
        return 1e6
    logit_=np.log(x)-np.log(1-x)
    return logit_

def get_matching_pairs(treated_df, non_treated_df, caliper):
    treated_df.index=range(len(treated_df))
    non_treated_df.index=range(len(non_treated_df))
    
    treated_x = treated_df.propensity_score_logit.values.reshape(-1,1)
    non_treated_x = non_treated_df.propensity_score_logit.values.reshape(-1,1)

    nbrs = NearestNeighbors(n_neighbors=1,n_jobs=-1, radius=caliper).fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x) #who are the males closest to these females?
    #since we find the closest male for each female, some males are selected more than once
    # print(distances)
    # print(indices)
    indices = indices.reshape(indices.shape[0]) #these are the indexes of the males
    matched_males = non_treated_df.iloc[indices]
    matched_males['counterfactual']=treated_df.PATIENT_ID.values
    treated_df['counterfactual']=matched_males.PATIENT_ID.values
    pairs=pd.concat([treated_df, matched_males]).sample(frac=1)# to shuffle the data
    return pairs

def load_pairs(filename,directory=config.DATAPATH):
    pairs=pd.DataFrame()
    t0=time()
    print('Loading ',filename)
    for chunk in pd.read_csv(filename, chunksize=100000):
        d = dict.fromkeys(chunk.columns, np.int8)
        d['PATIENT_ID']=np.int64
        d['counterfactual']=np.int64
        d['propensity_score']=np.float64
        d['propensity_score_logit']=np.float64
        chunk= chunk.astype(d)
        pairs = pd.concat([pairs, chunk], ignore_index=True)

    util.vprint('Loaded in ',time()-t0,' seconds')
    return(pairs)

#%%

np.random.seed(config.SEED)
no_duplicates=eval(input('Drop duplicates? (True/False) '))
plot=eval(input('Make plots? (True/False) '))
evaluate_matching=eval(input('Evaluate matching? (True/False) '))
#%%
X,y=getData(2016)
X,y=config.exclusion_criteria(X, y) if hasattr(config, 'exclusion_criteria') else X,y
original_columns=X.columns
print('Sample size ',len(X), ', female percentage: ',100*X.FEMALE.sum()/len(X))
assert not 'AGE_85GT' in X.columns
prefix='general_population_' if len(X)==2254252 else ''
print(prefix)
#%%
ps_model_name=f'{prefix}logistic_propensity_score_model'
filename=os.path.join(config.DATAPATH,f'{prefix}{config.EXPERIMENT}_pairs.csv')
ps_model_filename=os.path.join(config.MODELPATH,f'{ps_model_name}.joblib')
#%%
""" COMPUTE OR RELOAD PROPENSITY SCORE MODEL """
if (not Path(ps_model_filename).is_file()):
    logistic=LogisticRegression(penalty='none',max_iter=1000,verbose=0, n_jobs=-1)
    
    t0=time()
    fit=logistic.fit(X.drop(['FEMALE', 'PATIENT_ID'], axis=1), X['FEMALE'])
    print('fitting time: ',time()-t0)
    util.savemodel(config, fit,name=ps_model_name)
    
else:
    fit=job.load(ps_model_filename)
propensity_scores=fit.predict_proba(X.drop(['FEMALE', 'PATIENT_ID'], axis=1))[:,1]
propensity_logit = np.array([logit(xi) for xi in propensity_scores])

X.loc[:,'propensity_score'] = propensity_scores
X.loc[:,'propensity_score_logit'] = propensity_logit
X.loc[:,'outcome'] = y[config.COLUMNS].values
initial_AUC=roc_auc_score(X.FEMALE, propensity_scores)
print('Initial AUC for propensity scores: ', initial_AUC)
#%%
""" PERFORM MATCHING OR RELOAD PAIRS FROM FILE """
if not Path(filename).is_file():
   
    # X.loc[:,'propensity_score'] = propensity_scores
    # X.loc[:,'propensity_score_logit'] = propensity_logit
    # X.loc[:,'outcome'] = y.iloc[X.index][config.COLUMNS].values
    
    if plot:
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
        # Density distribution of propensity score (logic) broken down by treatment status
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Density distribution plots for propensity score')
        sns.kdeplot(x = propensity_scores, hue = X.FEMALE , ax = ax)
        ax.set_title('Propensity Score')
        
        plt.savefig(os.path.join(config.FIGUREPATH,f'{prefix}propensity_densities_before.png'))
        plt.show()
    
    """ MATCHING """
    
    common_support = (propensity_logit > -150) & (propensity_logit < 40)
    caliper = np.std(propensity_logit[common_support]) * 0.2
    
    print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))#radius is 0.2246
        
    pairs=get_matching_pairs(X.loc[X.FEMALE==1][list(original_columns)+['propensity_score','propensity_score_logit']],
                             X.loc[X.FEMALE==0][list(original_columns)+['propensity_score','propensity_score_logit']],
                             caliper)

    pairs.to_csv(filename, index=False)
    print('Saved ',filename)
else:
    pairs=load_pairs(filename)
    
if hasattr(config, 'exclusion_criteria'):
    assert not 'PHARMA_Benign_prostatic_hyperplasia' in pairs
#%

if no_duplicates:
    females_=pairs.loc[pairs.FEMALE==1].drop_duplicates('counterfactual')
    males_=pairs.loc[pairs.FEMALE==0].drop_duplicates(original_columns)
    pairs=pd.concat([males_, females_]).sample(frac=1)# to shuffle the data
    pairs.index=range(len(pairs))

pairs['outcome']=pd.merge(pairs[['PATIENT_ID']],y,on='PATIENT_ID')[config.COLUMNS]
#%%
threshold= 0 if config.ALGORITHM=='logistic' else y[config.COLUMNS].mean().values[0]
X['binary_outcome']=(y[config.COLUMNS]>threshold)
pairs['binary_outcome']=(pairs.outcome>threshold)
if evaluate_matching:
    print('Evaluating matches...')
    if plot:
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
        # Density distribution of propensity score (logic) broken down by treatment status
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Density distribution plots for propensity score.')
        sns.kdeplot(x = pairs.propensity_score.values, hue = pairs.FEMALE.values , ax = ax)
        ax.set_title('Propensity Score')  
        plt.savefig(os.path.join(config.FIGUREPATH,f'{prefix}two_propensity_densities_after.png'))
        plt.show()
        
        
        fig, ax = plt.subplots(2,1)
        fig.suptitle('Comparison of {} split by outcome and treatment status'.format('propensity_score_logit'))
        sns.stripplot(data = X, y = 'binary_outcome', alpha=0.5, x = 'propensity_score_logit', hue = 'FEMALE', orient = 'h', ax = ax[0]).set(title = 'Before matching', xlim=(-6, 4))
        sns.stripplot(data = pairs, y = 'binary_outcome', alpha=0.5, x = 'propensity_score_logit', hue = 'FEMALE', ax = ax[1] , orient = 'h').set(title = 'After matching', xlim=(-6, 4))
        plt.subplots_adjust(hspace = 0.3)
        plt.savefig(os.path.join(config.FIGUREPATH,f'{prefix}two_stripplot.png'))
        plt.show()
        
        sns.set(rc={'figure.figsize':(10,60)}, font_scale=1.0)
        def cohenD (tmp, metricName):
            treated_metric = tmp[tmp.FEMALE == 1][metricName]
            untreated_metric = tmp[tmp.FEMALE == 0][metricName]
            
            d = ( treated_metric.mean() - untreated_metric.mean() ) / np.sqrt(((treated_metric.count()-1)*treated_metric.std()**2 + (untreated_metric.count()-1)*untreated_metric.std()**2) / (treated_metric.count() + untreated_metric.count()-2))
            return d
        data = []
        cols = [c for c in original_columns if not ((c=='FEMALE') or (c=='PATIENT_ID'))]
        for cl in cols:
            data.append([cl,'before', cohenD(X,cl)])
            data.append([cl,'after', cohenD(pairs,cl)])
        
        res = pd.DataFrame(data, columns=['CATEGORIES','matching','effect_size'])
        descriptions=pd.read_csv(config.DATAPATH+'CCSCategoryNames_FullLabels.csv')
        res=pd.merge(res,descriptions,on='CATEGORIES',how='left')
        res['variable']=np.where(res.LABELS.isna(),res.CATEGORIES,res.LABELS)
        
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(10,60)}, font_scale=1.0)
        sn_plot = sns.barplot(data = res, y = 'variable', x = 'effect_size', hue = 'matching', orient='h')
        sn_plot.set(title='Standardised Mean differences accross covariates before and after matching')
        sn_plot.figure.savefig(os.path.join(config.FIGUREPATH,f"{prefix}two_all_standardised_mean_differences.png"))
        
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(10,60)}, font_scale=1.0)
        sn_plot = sns.barplot(data = res, y = 'variable', x = res.effect_size.abs(), hue = 'matching', orient='h')
        sn_plot.set(title='Absolute value of standardised mean differences accross covariates before and after matching')
        sn_plot.figure.savefig(os.path.join(config.FIGUREPATH,f"{prefix}two_all_abs_standardised_mean_differences.png"))
    
    #%%
    # Evaluate the decrease in AUC
    logistic=LogisticRegression(penalty='none',max_iter=400,n_jobs=-1)
    newfit=logistic.fit(pairs[original_columns].drop(['FEMALE','PATIENT_ID'], axis=1), pairs.FEMALE.values.reshape(-1,1))
    new_propensity_scores=newfit.predict_proba(pairs[original_columns].drop(['FEMALE','PATIENT_ID'], axis=1))[:,1]
    new_AUC=roc_auc_score(pairs.FEMALE, new_propensity_scores)
    #%%
    print('Initial AUC for propensity scores: ', initial_AUC)
    print('After matching, AUC for new propensity scores: ', new_AUC)
#%%
overview = pairs[['outcome','FEMALE']].groupby(by = ['FEMALE']).aggregate([np.mean, np.var, np.std, 'count'])
print(overview)

treated_outcome = overview['outcome']['mean'][1]
treated_counterfactual_outcome = overview['outcome']['mean'][0]
att = treated_outcome - treated_counterfactual_outcome
print('The Average Treatment Effect in females is (ATT): {:.4f}'.format(att))
#Vemos que, en una muestra de mujeres y hombres con las mismas 
#características clínicas, las mujeres ingresan menos (ATT<0)
if config.ALGORITHM=='linear':
    dfWomen=pairs[['outcome','FEMALE']].loc[pairs.FEMALE==1]
    dfMen=pairs[['outcome','FEMALE']].loc[pairs.FEMALE==0]
    import statsmodels.stats.api as sms
    r = sms.CompareMeans(sms.DescrStatsW(dfWomen.outcome),
                         sms.DescrStatsW(dfMen.outcome))
    
   
    print('Confidence interval for the ATT; ',r.tconfint_diff())

#%%
overview2 = pairs[['binary_outcome','FEMALE']].groupby(by = ['FEMALE']).aggregate(['sum',np.mean, np.var, np.std, 'count'])
print(overview2.to_markdown())
treated_outcome = overview2['binary_outcome']['mean'][1]
treated_counterfactual_outcome = overview2['binary_outcome']['mean'][0]
att2 = treated_outcome - treated_counterfactual_outcome
print('For presence/absence of outcome, the Average Treatment Effect in females is (ATT): {:.4f}'.format(att2))
from statsmodels.stats.proportion import proportion_confint 

if config.ALGORITHM=='logistic':
    confint_prevalence_women=proportion_confint(count=overview2['binary_outcome']['sum'][1],    # Number of "successes"
                       nobs=len(pairs)/2,    # Number of trials
                       alpha=(1 - 0.95))
    
    confint_prevalence_men=proportion_confint(count=overview2['binary_outcome']['sum'][0],    # Number of "successes"
                       nobs=len(pairs)/2,    # Number of trials
                       alpha=(1 - 0.95))
    
    print('Confidence interval for the ATT; ',np.array(confint_prevalence_women)-np.array(confint_prevalence_men))
#%%
ypairs=pairs.outcome
x=pairs[original_columns].drop(['PATIENT_ID'],axis=1)

ypairs=np.where(ypairs>=1,1,0) if config.ALGORITHM=='logistic' else ypairs
ypairs=ypairs.ravel()
#%%
if config.ALGORITHM=='logistic':
    estimator=LogisticRegression(penalty='none',max_iter=1000,verbose=0,n_jobs=-1)
elif config.ALGORITHM=='linear':
    estimator=LinearRegression(n_jobs=-1)  
if not Path(os.path.join(config.MODELPATH,f'{prefix}{config.ALGORITHM}_psm.joblib')).is_file():    
    print('fitting ...')
    t0=time()
    fit=estimator.fit(x, ypairs)
    print('fitting time: ',time()-t0)
    util.savemodel(config, fit,name=f'{prefix}{config.ALGORITHM}_psm')
else:
    fit=job.load(os.path.join(config.MODELPATH,f'{prefix}{config.ALGORITHM}_psm.joblib'))
if config.ALGORITHM=='logistic':
    predict=fit.predict_proba
    score=roc_auc_score
    score2=average_precision_score
    kwargs_score2={}
    score_name,score2_name='AUC','AP'
elif config.ALGORITHM=='linear':
    predict=fit.predict
    score=r2_score
    score2=mean_squared_error
    kwargs_score2={'squared':False}
    score_name,score2_name='R2','RMSE' 
#%%
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
def plot_pr(precision, recall, groupname,  y , pred):
    avg_prec = average_precision_score(y, pred)
    display = PrecisionRecallDisplay(precision=precision, recall=recall,
                                     estimator_name=groupname,
                                     average_precision=avg_prec)
    return display
if config.ALGORITHM=='logistic':
    yMale=np.where(pairs.loc[pairs.FEMALE==0].outcome>=1,1,0)
    yFemale=np.where(pairs.loc[pairs.FEMALE==1].outcome>=1,1,0)
else:
    yMale=pairs.loc[pairs.FEMALE==0].outcome
    yFemale=pairs.loc[pairs.FEMALE==1].outcome
predsMale=predict(x.loc[x.FEMALE==0])
predsFemale=predict(x.loc[x.FEMALE==1])

if config.ALGORITHM=='linear':
    allpreds=pd.concat([pd.DataFrame({'PRED':predsMale,'Sex':[0]*len(predsMale)}),
                    pd.DataFrame({'PRED':predsFemale,'Sex':[1]*len(predsFemale)})])
else:
    allpreds=pd.concat([pd.DataFrame({'PRED':predsMale[:,1],'Sex':[0]*len(predsMale)}),
                    pd.DataFrame({'PRED':predsFemale[:,1],'Sex':[1]*len(predsFemale)})])
#%%
#Number of females in the top 20000 list
K=20000
percent=allpreds.nlargest(K,'PRED').Sex.sum()*100/K
print(f'The percentage of women in the top {K} list is ',percent)
#%%
from matplotlib import pyplot as plt
recall, PPV= {}, {}
plot={}
sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
fig, ax = plt.subplots(1,1)
for sex, yy, preds in zip(['Male', 'Female'], [yMale, yFemale], [predsMale, predsFemale]):
    if config.ALGORITHM=='logistic':
        preds=preds[:,1]
    print(f'{score_name} {sex}= ',score(yy,preds))
    print(f'{score2_name} {sex}= ',score2(yy,preds, **kwargs_score2))
    rec, ppv, _, _=performance(yy, preds, K=20000)
    print(f'Recall_20000 {sex}= ', rec)
    print(f'PPV_20000 {sex}= ', ppv)
    print(' ')
    if config.ALGORITHM=='logistic':
        
        # 
        prec, rec, thre = precision_recall_curve(yy, preds)
        plot[sex]=plot_pr(prec, rec, sex, yy, preds).plot(ax)
    if config.ALGORITHM=='linear':
        recall[sex], PPV[sex]=[],[]
        for K in range(1000,len(preds), 10000):
            rec, ppv, _, _=performance(yy, preds, K=K, verbose=False)
            recall[sex].append(rec) ; PPV[sex].append(ppv)
#%%
import seaborn as sns
if config.ALGORITHM=='logistic':
    for sex in ['Male', 'Female']:
        sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
        plot[sex].plot(ax)
if config.ALGORITHM=='linear':
    import seaborn as sns
    fig, (ax1, ax2) = plt.subplots(1,2)
    # for sex in ['Male', 'Female']:  
    ax1.plot(recall['Male'], recall['Female'])
    ax1.set_ylabel('Male') ; ax1.set_xlabel('Female') ; ax1.set_title('Recall')
    ax2.plot(PPV['Male'], PPV['Female']) ; ax2.set_title('PPV')
    
    for ax in (ax1,ax2):
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.tight_layout()
#%%
coefs=fit.coef_[0] if config.ALGORITHM=='logistic' else fit.coef_
betas={k:v for k,v in zip(fit.feature_names_in_,coefs)}
print('Coefficient for females: ',betas['FEMALE'])
print('Intercept: ',fit.intercept_)
#%%
""" ASSESSING SIGNIFICANCE OF BETA FOR FEMALES 
With all the predictors we have a non-invertible matrix, and the variance 
of the coefficients is inflated because we are computing a pseudoinverse.

Luckily, after matching, we have removed most of the association between 
sex and the rest of the predictors. Thus, if we fit a univariate model with
sex as the independent variable, we will obtain a similar beta coefficient,
and we will be able to estimate its variance and statistical significance.

If the coefficient for sex is statistically significant, so is the size 
of the effect.
"""
import statsmodels.api as sm
independent = sm.add_constant(x['FEMALE'], prepend=False)
if config.ALGORITHM=='linear':
    mod = sm.OLS(ypairs, independent)
else:
    mod = sm.Logit(ypairs, independent)
res = mod.fit()
print(res.summary())
