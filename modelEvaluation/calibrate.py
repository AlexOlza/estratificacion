# IMPORTS FROM EXTERNAL LIBRARIES
import os
from pathlib import Path
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
from modelEvaluation.predict import predict, generate_filename
from modelEvaluation.compare import detect_models, detect_latest
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
    calibFilename=generate_filename(model_name,yr, calibrated=True)
    if Path(calibFilename).is_file():
        util.vprint('Calibrated predictions found; loading')
        p_calibrated=pd.read_csv(calibFilename)
        return(p_calibrated)
    pastX=kwargs.get('pastX',None)
    pastY=kwargs.get('pastY',None)
    presentX=kwargs.get('presentX',None)
    presentY=kwargs.get('presentY',None)
    if (not isinstance(pastX,pd.DataFrame)) or (not isinstance(pastY,pd.DataFrame)):
        pastX,pastY=getData(yr-2)
    if (not isinstance(presentX,pd.DataFrame)) or (not isinstance(presentY,pd.DataFrame)):
        presentX,presentY=getData(yr-1)

    #This reads the prediction files, or creates them if not present
    pastPred, _= predict(model_name,config.EXPERIMENT,yr-1, X=pastX, y=pastY)
    pred, _= predict(model_name,config.EXPERIMENT,yr, X=presentX, y=presentY)
    
    pastPred.sort_values(by='PATIENT_ID',inplace=True)
    pastPred.reset_index(drop=True,inplace=True)
    
    p_train, _ , y_train, _ = train_test_split(pastPred.PRED.values, np.where(pastY[config.COLUMNS]>=1,1,0).ravel(),
                                                        test_size=0.33, random_state=config.SEED)
    
    ir = IsotonicRegression( out_of_bounds = 'clip' )	
    ir.fit( p_train, y_train )
    
    pred.sort_values(by='PATIENT_ID',inplace=True)
    pred.reset_index(drop=True,inplace=True)
    p_uncal=pred.PRED
    p_calibrated_iso = ir.transform( p_uncal )
    util.vprint('Number of unique probabilities after isotonic regression: ',len(set(p_calibrated_iso )))
    
    idx,p_sample=sample(p_calibrated_iso,p_uncal)
    p_uncal_sample=p_uncal[idx]

    pchip=PchipInterpolator(p_uncal_sample.values, p_sample, axis=0, extrapolate=True) 
    p_calibrated=pchip(p_uncal)
    pred['PRED']=p_calibrated
    util.vprint('Number of unique probabilities after PCHIP: ',len(set(p_calibrated)))
    pred.to_csv(calibFilename)
    util.vprint('Saved ',calibFilename)
    return(pred)

def plot():
    pass
    # for p,name in zip([p_uncal,p_calibrated],[n+'',n+' Calibrated']):
    #     fraction_of_positives, mean_pastPredicted_value = \
    #             calibration_curve(y_test, p, n_bins=10,normalize=False)
        
        
       
    #     if name==n+' Calibrated':
    #         unique=len(np.unique(p_calibrated))
            
            
    #         bintot=bin_total(y_test, p_calibrated, n_bins=10)
    #         notempty=[i for i in range(len(mean_pastPredicted_value)) if bintot[i] != 0]
    #         mean_pastPredicted_valuex=[mean_pastPredicted_value[i] for i in notempty]
    #         fraction_of_positivesx=[fraction_of_positives[i] for i in notempty]
    #         ax1.plot(mean_pastPredicted_valuex, fraction_of_positivesx, "s-",color=col,
    #               label="%s" % (name, ))
    #         for i, txt in zip(range(len(mean_pastPredicted_value)),bintot):
    #                 if i>len(mean_pastPredicted_value)-3 and txt!=0:
    #                     ax1.annotate(txt, (mean_pastPredicted_value[i], fraction_of_positives[i]),fontsize=medium)
    #                     print(i,txt)
    #         df=pd.DataFrame()
    #         df['Fracción positivos']=fraction_of_positives
    #         df['pastPredicción media']=mean_pastPredicted_value
    #         df['N']=bintot[bintot!=0]
    #         print('\n')
    #         # print(df.to_latex(float_format="%.3f",index=False,caption='Tabla de calibrado para {0} con regresión isotónica y PCHIP'.format(n)))
    #         # print('\n')
    #         # reliabilityConsistency(p_calibrated, y_test, nbins=10, nboot=100, ax=ax1, seed=42,color=col)
            
    #         # ax2.hist(p, range=(0, 1), bins=10, label=n,
    #         #             histtype="step", lw=2)
            
    #         # print('unique ',unique)
    #         ax3.hist(p, range=(0, 1), bins=unique,
    #                 histtype="step", lw=2,color=col)
    #     # else:
    #     #     print('unique ',unique)
    #     #     ax2.plot(mean_pastPredicted_value, fraction_of_positives, "s-",color=col,
    #     #           label="%s" % (name, ))
    #     #     ax4.hist(p, range=(0, 1), bins=unique, label=n,
    #     #             histtype="step", lw=2,color=col)
    # # ax1.annotate('123',(0.6,0.6),fontsize=medium)
    # ax1.set_ylabel("Fraction of positives")
    # ax1.set_xlabel("Mean pastPredicted value")
    # ax1.set_ylim([-0.05, 1.05])
    # ax1.legend(loc="upper center", ncol=2)
    # # ax1.set_title('Reliability diagram after calibration')
    
    # # ax2.set_xlabel("Mean pastPredicted value")
    # # ax2.legend(loc="upper center", ncol=2)
    # # ax2.set_title('Before Calibration')
    
    # ax3.set_ylabel("Count")
    # ax3.set_xlabel("pastPredicted probability")
    # ax3.ticklabel_format(style='sci',axis='both')
    # plt.tight_layout()
    # # plt.savefig('pastPredecirIngresos/reliabilityConsistency/{0}IsoPCHIP.eps'.format(n),format='eps')
    # plt.show()


#%%
if __name__=='__main__':
        
    year=int(input('YEAR YOU WANT TO CALIBRATE:'))
    model_name=input('MODEL NAME (example: logistic20220118_132612): ')
    
    pastX,pastY=getData(year-2)
    presentX,presentY=getData(year-1)
    models=detect_latest(detect_models())
    p={}
    for model_name in models:
        p[model_name]=calibrate(model_name,year,
                pastX=pastX,pastY=pastY,presentX=presentX,presentY=presentY)
 
    # plot