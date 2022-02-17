# IMPORTS FROM EXTERNAL LIBRARIES
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import PchipInterpolator
#%%
#IMPORTS FROM THIS PROJECT
from python_settings import settings as config
import configurations.utility as util
from modelEvaluation.pastPredict import pastPredict, generate_filename
from modelEvaluation.compare import detect_models 
util.configure('configurations.local.logistic')
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
    pastX=kwargs.get('pastX',None)
    pastY=kwargs.get('pastY',None)
    presentX=kwargs.get('presentX',None)
    presentY=kwargs.get('presentY',None)
    if (not isinstance(pastX,pd.DataFrame)) or (not isinstance(pastY,pd.DataFrame)):
        pastX,pastY=getData(yr-1)
    if (not isinstance(presentX,pd.DataFrame)) or (not isinstance(presentY,pd.DataFrame)):
        presentX,presentY=getData(yr)

    pastPredFile=generate_filename(model_name,yr-1)
    predFile=generate_filename(model_name,yr)
    
    pastPred=pd.read_csv(pastPredFile)
    pastPred.sort_values(by='PATIENT_ID',inplace=True)
    pastPred.reset_index(drop=True,inplace=True)
    print(pastPred.columns)
    
    p_train, p_test, y_train, y_test = train_test_split(pastPred[pastPred.columns[1]], pastData[config.COLUMNS],
                                                      test_size=0.33, random_state=42)
    
    ir = IsotonicRegression( out_of_bounds = 'clip' )	
    ir.fit( p_train, y_train )
    print('fitted ir')
    pred=pd.read_csv(predFile)
    pred.sort_values(by='PATIENT_ID',inplace=True)
    pred.reset_index(drop=True,inplace=True)
    p_uncal=pred[pred.columns[1]]
    p_calibrated_iso = ir.transform( p_uncal )
    print('num unique probs ',len(set(p_calibrated_iso )))
    print('ir transformed')
    idx,p_sample=sample(p_calibrated_iso,p_uncal)
    print('sample drawn')
    y_test=presentData[config.COLUMNS]
    y_sample=y_test[list(idx)]
    p_uncal_sample=p_uncal[idx]
    print(len(p_sample),len(y_sample))
    pchip=PchipInterpolator(p_uncal_sample.values, p_sample, axis=0, extrapolate=True) 
    p_calibrated=pchip(p_uncal)
    return(p_calibrated)#OR MAYBE NO RETURN

def plot():
    
    for p,name in zip([p_uncal,p_calibrated],[n+'',n+' Calibrated']):
        fraction_of_positives, mean_pastPredicted_value = \
                calibration_curve(y_test, p, n_bins=10,normalize=False)
        
        
       
        if name==n+' Calibrated':
            unique=len(np.unique(p_calibrated))
            
            
            bintot=bin_total(y_test, p_calibrated, n_bins=10)
            notempty=[i for i in range(len(mean_pastPredicted_value)) if bintot[i] != 0]
            mean_pastPredicted_valuex=[mean_pastPredicted_value[i] for i in notempty]
            fraction_of_positivesx=[fraction_of_positives[i] for i in notempty]
            ax1.plot(mean_pastPredicted_valuex, fraction_of_positivesx, "s-",color=col,
                  label="%s" % (name, ))
            for i, txt in zip(range(len(mean_pastPredicted_value)),bintot):
                    if i>len(mean_pastPredicted_value)-3 and txt!=0:
                        ax1.annotate(txt, (mean_pastPredicted_value[i], fraction_of_positives[i]),fontsize=medium)
                        print(i,txt)
            df=pd.DataFrame()
            df['Fracción positivos']=fraction_of_positives
            df['pastPredicción media']=mean_pastPredicted_value
            df['N']=bintot[bintot!=0]
            print('\n')
            # print(df.to_latex(float_format="%.3f",index=False,caption='Tabla de calibrado para {0} con regresión isotónica y PCHIP'.format(n)))
            # print('\n')
            # reliabilityConsistency(p_calibrated, y_test, nbins=10, nboot=100, ax=ax1, seed=42,color=col)
            
            # ax2.hist(p, range=(0, 1), bins=10, label=n,
            #             histtype="step", lw=2)
            
            # print('unique ',unique)
            ax3.hist(p, range=(0, 1), bins=unique,
                    histtype="step", lw=2,color=col)
        # else:
        #     print('unique ',unique)
        #     ax2.plot(mean_pastPredicted_value, fraction_of_positives, "s-",color=col,
        #           label="%s" % (name, ))
        #     ax4.hist(p, range=(0, 1), bins=unique, label=n,
        #             histtype="step", lw=2,color=col)
    # ax1.annotate('123',(0.6,0.6),fontsize=medium)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Mean pastPredicted value")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper center", ncol=2)
    # ax1.set_title('Reliability diagram after calibration')
    
    # ax2.set_xlabel("Mean pastPredicted value")
    # ax2.legend(loc="upper center", ncol=2)
    # ax2.set_title('Before Calibration')
    
    ax3.set_ylabel("Count")
    ax3.set_xlabel("pastPredicted probability")
    ax3.ticklabel_format(style='sci',axis='both')
    plt.tight_layout()
    # plt.savefig('pastPredecirIngresos/reliabilityConsistency/{0}IsoPCHIP.eps'.format(n),format='eps')
    plt.show()


#%%
if __name__=='__main__':
        
    year=int(input('YEAR YOU WANT TO CALIBRATE:'))
    detect all pastPrediction filenames for this experiment
    detect all calibrated pastPrediction filenames for this experiment
    calibrate those that are missing
    plot
