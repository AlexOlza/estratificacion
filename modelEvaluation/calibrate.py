#CONFIGURE
import os
import numpy as np
from predecirIngresos.calibradoIntervalos import reliabilityConsistency
np.random.seed(0)
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import PchipInterpolator

REQUEST INPUT: yr
datos17, datos18 = getData(yr-1), getData(yr)


#%%    
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

for f17,f18,n,col in zip(files17,files18,names,colors):
    # if n!= 'AdaBoost Trees':
    #     continue
    print(n)
    pred=pd.read_csv(f17)
    pred.sort_values(by='PATIENT_ID',inplace=True)
    pred.reset_index(drop=True,inplace=True)
    
    p_train, p_test, y_train, y_test = train_test_split(pred[pred.columns[1]], datos17['algunIngresoUrg'],
                                                      test_size=0.33, random_state=42)
    
    ir = IsotonicRegression( out_of_bounds = 'clip' )	
    ir.fit( p_train, y_train )
    print('fitted ir')
    pred18=pd.read_csv(f18)
    pred18.sort_values(by='PATIENT_ID',inplace=True)
    pred18.reset_index(drop=True,inplace=True)
    p_uncal=pred18[pred18.columns[1]]
    p_calibrated_iso = ir.transform( p_uncal )
    print('num unique probs ',len(set(p_calibrated_iso )))
    print('ir transformed')
    idx,p_sample=sample(p_calibrated_iso,p_uncal)
    print('sample drawn')
    y_test=datos18.algunIngresoUrg
    y_sample=y_test[list(idx)]
    p_uncal_sample=p_uncal[idx]
    print(len(p_sample),len(y_sample))
    # p_sampletrans=ir.transform( p_sample)
    pchip=PchipInterpolator(p_uncal_sample.values, p_sample, axis=0, extrapolate=True) 
    p_calibrated=pchip(p_uncal)
    # print('number of probabilities out of range=',len(p_calibrated[p_calibrated<0])+len(p_calibrated[p_calibrated>1]))

    for p,name in zip([p_uncal,p_calibrated],[n+'',n+' Calibrated']):
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, p, n_bins=10,normalize=False)
        
        
       
        if name==n+' Calibrated':
            unique=len(np.unique(p_calibrated))
            
            
            bintot=bin_total(y_test, p_calibrated, n_bins=10)
            notempty=[i for i in range(len(mean_predicted_value)) if bintot[i] != 0]
            mean_predicted_valuex=[mean_predicted_value[i] for i in notempty]
            fraction_of_positivesx=[fraction_of_positives[i] for i in notempty]
            ax1.plot(mean_predicted_valuex, fraction_of_positivesx, "s-",color=col,
                  label="%s" % (name, ))
            for i, txt in zip(range(len(mean_predicted_value)),bintot):
                    if i>len(mean_predicted_value)-3 and txt!=0:
                        ax1.annotate(txt, (mean_predicted_value[i], fraction_of_positives[i]),fontsize=medium)
                        print(i,txt)
            df=pd.DataFrame()
            df['Fracción positivos']=fraction_of_positives
            df['Predicción media']=mean_predicted_value
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
        #     ax2.plot(mean_predicted_value, fraction_of_positives, "s-",color=col,
        #           label="%s" % (name, ))
        #     ax4.hist(p, range=(0, 1), bins=unique, label=n,
        #             histtype="step", lw=2,color=col)
# ax1.annotate('123',(0.6,0.6),fontsize=medium)
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="upper center", ncol=2)
# ax1.set_title('Reliability diagram after calibration')

# ax2.set_xlabel("Mean predicted value")
# ax2.legend(loc="upper center", ncol=2)
# ax2.set_title('Before Calibration')

ax3.set_ylabel("Count")
ax3.set_xlabel("Predicted probability")
ax3.ticklabel_format(style='sci',axis='both')
plt.tight_layout()
# plt.savefig('predecirIngresos/reliabilityConsistency/{0}IsoPCHIP.eps'.format(n),format='eps')
plt.show()

    
