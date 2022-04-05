
# IMPORTS FROM EXTERNAL LIBRARIES
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import brier_score_loss
#%%
#IMPORTS FROM THIS PROJECT
from python_settings import settings as config
import configurations.utility as util
if not config.configured:
    config_used=os.path.join(os.environ['USEDCONFIG_PATH'],input('Experiment...'),input('Model...')+'.json')
    configuration=util.configure(config_used)

from modelEvaluation.predict import predict, generate_filename
from modelEvaluation.compare import detect_models, detect_latest
from dataManipulation.dataPreparation import getData
from modelEvaluation.reliableDiagram import reliabilityConsistency
np.random.seed(config.SEED)

#%%    
#FUNCTIONS
def bin_total(y_true, y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1

    return np.bincount(binids, minlength=len(bins))[:-1]
def same(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def sample(data,uncal):
    unique=np.unique(data)
    datalist=data.tolist()
    idx=[datalist.index(s) for s in unique[1:-1]]
    idx.insert(0,np.argmin(uncal))
    idx.append(np.argmax(uncal))
    return(idx,unique)

def calibrate(model_name,yr,**kwargs):
    try:
        filename=kwargs.get('filename',model_name)
        experiment_name=kwargs.get('experiment_name',config.EXPERIMENT)
        calibFilename=generate_filename(filename,yr, calibrated=True)
        uncalFilename=generate_filename(filename,yr, calibrated=False)
        if Path(calibFilename).is_file():
            util.vprint('Calibrated predictions found; loading')
            p_calibrated=pd.read_csv(calibFilename)
            return(p_calibrated)
        predictors=kwargs.get('predictors',config.PREDICTORREGEX)
        pastX=kwargs.get('pastX',None)
        pastY=kwargs.get('pastY',None)
        presentX=kwargs.get('presentX',None)
        presentY=kwargs.get('presentY',None)
        if (not isinstance(pastX,pd.DataFrame)) or (not isinstance(pastY,pd.DataFrame)):
            pastX,pastY=getData(yr-2)
        if (not isinstance(presentX,pd.DataFrame)) or (not isinstance(presentY,pd.DataFrame)):
            presentX,presentY=getData(yr-1)
    
        #This reads the prediction files, or creates them if not present
        pastPred, _= predict(model_name,experiment_name,yr-1,
                             filename=filename,
                             X=pastX, y=pastY, predictors=predictors)
        pred, _= predict(model_name,experiment_name,yr,
                         filename=filename,
                         X=presentX, y=presentY, predictors=predictors)
        
        pastPred.sort_values(by='PATIENT_ID',inplace=True)
        pastPred.reset_index(drop=True,inplace=True)
        
        p_train, _ , y_train, _ = train_test_split(pastPred.PRED.values, np.where(pastPred.OBS>=1,1,0).ravel(),
                                                            test_size=0.33, random_state=config.SEED)
        
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds = 'clip' )	
        ir.fit( p_train, y_train )
        
        pred.sort_values(by='PATIENT_ID',inplace=True)
        pred.reset_index(drop=True,inplace=True)
        p_uncal=pred.PRED
        p_calibrated_iso = ir.transform( p_uncal )
        util.vprint('Number of unique probabilities after isotonic regression: ',len(set(p_calibrated_iso )))
        
        idx,p_sample=sample(p_calibrated_iso,p_uncal)
        p_uncal_sample=p_uncal[idx]
    
        pchip=PchipInterpolator(p_uncal_sample.values, p_sample, axis=0, extrapolate=False) 
        p_calibrated=pchip(p_uncal)
        
        util.vprint('Number of invalid probabilities (set to 1): ',sum(p_calibrated>1))
        util.vprint('Number of invalid probabilities (set to 0): ',sum(p_calibrated<0))
        util.vprint('Number of unique probabilities after PCHIP: ',len(set(p_calibrated)))
        p_calibrated[(p_calibrated>1)]=1
        p_calibrated[(p_calibrated<0)]=0
        pred['PREDCAL']=p_calibrated
        pred.to_csv(calibFilename,index=False)
        util.vprint('Saved ',calibFilename)
        uncalFilenames=[generate_filename(filename,yr, calibrated=False),
                       generate_filename(filename,yr-1, calibrated=False)]
        for f in uncalFilenames:
            if Path(f).is_file():
                Path(f).unlink()
                print(f'Deleted {f}')
        return(pred)
    except:
        return None
def plot(p, **kwargs):
    path=kwargs.get('path',config.FIGUREPATH)
    filename=kwargs.get('filename','')
    util.makeAllPaths()
    from matplotlib.gridspec import GridSpec
    names=list(p.keys())
    fig=plt.figure(1,figsize=(15, 10))
    fig2=plt.figure(2,figsize=(15, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1])
    gs2 = GridSpec(2, 1,figure=fig2,height_ratios=[4, 1])
    ax2 = fig2.add_subplot(gs2[0])
    ax1 = fig.add_subplot(gs[0])
    ax4 = fig2.add_subplot(gs2[1])
    ax3 = fig.add_subplot(gs[1])
    ax1.plot([0, 1], [0, 1], "k:", label=0)
    ax2.plot([0, 1], [0, 1], "k:", label=0)
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd']#, '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    large=27
    medium=18
    for model_name, preds, col in zip(p.keys(),p.values(),colors):
        obs=np.where(preds.OBS>=1,1,0).ravel()
        for probs,name in zip([preds.PRED,preds.PREDCAL],[model_name,model_name+' Calibrated']):
            fraction_of_positives, mean_pastPredicted_value = \
                    calibration_curve(obs, probs, n_bins=20,normalize=False)
            
            if 'Calibrated' not in name:
                axTop=ax1
                axHist=ax3
            else:
                axTop=ax2
                axHist=ax4
            unique=len(np.unique(probs))
       
            bintot=bin_total(obs, probs, n_bins=20)
            label=brier_score_loss(obs, probs)
            print('brier ',label)
            notempty=[i for i in range(len(mean_pastPredicted_value)) if bintot[i] != 0]
            mean_pastPredicted_valuex=[mean_pastPredicted_value[i] for i in notempty]
            fraction_of_positivesx=[fraction_of_positives[i] for i in notempty]
            axTop.plot(mean_pastPredicted_valuex, fraction_of_positivesx, "s-",color=col,
                  label="%s" % (label, ))
            for i, txt in zip(range(len(mean_pastPredicted_value)),bintot):
                    if i>len(mean_pastPredicted_value)-3 and txt!=0:
                        axTop.annotate(txt, (mean_pastPredicted_value[i], fraction_of_positives[i]),fontsize=medium)
                        print(i,txt)
            df=pd.DataFrame()
            df['Fracción positivos']=fraction_of_positives
            df['pastPredicción media']=mean_pastPredicted_value
            df['N']=bintot[bintot!=0]
            print('\n')


            reliabilityConsistency(probs, obs, nbins=20, nboot=100, ax=axTop, seed=config.SEED,color=col)
            
            axHist.hist(probs, range=(0, 1), bins=unique,
                    histtype="step", lw=2,color=col)

            axTop.set_ylabel("Fraction of positives",fontsize=medium)
            axTop.set_xlabel("Mean predicted value",fontsize=medium)
            axTop.set_ylim([-0.05, 1.10])
            axTop.legend(loc="upper center", ncol=2,fontsize=medium,title='Brier score')
            axTop.get_legend().get_title().set_fontsize(medium)
            axHist.set_ylabel("Count",fontsize=medium)
            axHist.set_xlabel("Predicted probability",fontsize=medium)
    ax1.set_title('Before calibration',fontsize=large)
    ax2.set_title('After Calibration',fontsize=large)
    
    handles, labels = ax2.get_legend_handles_labels()
    for f in [fig,fig2]:
        f.legend(handles, ['Perfectly calibrated']+names,shadow=True, loc='lower center',ncol=3,fontsize=medium,bbox_to_anchor=(0.5,-0.06))
        f.tight_layout(rect=[0, 1, 1, 0.95],w_pad=4.0)
    gs.tight_layout(fig)
    gs2.tight_layout(fig2)
    plt.show()
    fig.savefig(os.path.join(path,filename+'BeforeCal.png'), bbox_inches = 'tight')
    fig2.savefig(os.path.join(path,filename+'AfterCal.png'), bbox_inches = 'tight')
    

#%%
if __name__=='__main__':
        
    year=int(input('YEAR YOU WANT TO CALIBRATE:'))

    pastX,pastY=getData(year-2)
    presentX,presentY=getData(year-1)
    models=detect_models()
    p={}
    brier_before,brier_after={},{}
    for model_name in models:
        print(model_name)
        try:
            # p[model_name]=calibrate(model_name,year,
            #         pastX=pastX,pastY=pastY,presentX=presentX,presentY=presentY)
            brier_before[model_name]=brier_score_loss(np.where(p[model_name].OBS>=1,1,0), p[model_name].PRED)
            brier_after[model_name]=brier_score_loss(np.where(p[model_name].OBS>=1,1,0), p[model_name].PREDCAL)
        except:
            print('SOMETHING WENT WRONG')
            p[model_name]=calibrate(model_name,year,
                    pastX=pastX,pastY=pastY,presentX=presentX,presentY=presentY)
            brier_before[model_name]=brier_score_loss(np.where(p[model_name].OBS>=1,1,0), p[model_name].PRED)
            brier_after[model_name]=brier_score_loss(np.where(p[model_name].OBS>=1,1,0), p[model_name].PREDCAL)
    
    # For each algorithm, select the models with the 25th percentile, median and 75th percentile of Brier score
    algorithms=list(set(['_'.join(re.findall('[^\d+_\d+]+',model)) for model in  p.keys()]))
    
    brier_after_alg=pd.DataFrame({'Model':brier_after.keys(),
                     'Brier':brier_after.values(),
                     'Algorithm':['_'.join(re.findall('[^\d+_\d+]+',model)) for model in  p.keys()]}
                                 )
    
    def q1(x):
        return x.quantile(0.25,interpolation='nearest')

    def q2(x):
        return x.quantile(0.5,interpolation='nearest')
    
    def q3(x):
        return x.quantile(0.75,interpolation='nearest')

    brierdist=brier_after_alg.groupby(['Algorithm']).Brier.agg([ q1, q2, q3]).stack(level=0)
    print('Brier score distribution per algorithm: ')
    print(brierdist)

    low, median, high = {},{},{}
    selected_models=[]
    for alg in algorithms:
        perc25=brierdist.loc[alg]['q1']
        perc50=brierdist.loc[alg]['q2']
        perc75=brierdist.loc[alg]['q3']
        low[alg]=list(brier_after.keys())[list(brier_after.values()).index(perc25)]
        median[alg]=list(brier_after.keys())[list(brier_after.values()).index(perc50)]
        high[alg]=list(brier_after.keys())[list(brier_after.values()).index(perc75)]
        selected_models+=[low[alg],median[alg],high[alg]]
    models_to_plot={k: v for k, v in p.items() if k in selected_models}

    plot(models_to_plot)
