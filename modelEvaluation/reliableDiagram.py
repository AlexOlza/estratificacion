#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFERENCE: Increasing the Reliability of Reliability Diagrams, BRÖCKER AND SMITH
Created on Thu Aug 19 16:11:22 2021

@author: alexander olza
INPUT:
p: Probability estimates
y: Actual outcomes
ax: A matplotlib axes, where you should plot the reliability diagram.
This function will add the consistency bars.
"""
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice, uniform
def reliabilityConsistency(p, y, nbins, nboot, ax, color='k', verbose=False, seed=None):
    #tip: assert len p = len y else raise error
    if seed:
        np.random.seed(seed)
    N=len(p)
    mean=[]
    pos=[]
    xtuples,ytuples=[],[]
    for i in range(nboot):
        if verbose:
            print(i,'/',nboot)
        surrogateForecasts=choice(p, size=N)
        unif=uniform(size=N)
        surrogateVerifications=[(1 if (unif[i]<surrogateForecasts[i]) else 0) for i in range(N)]
        fraction_of_positives, mean_predicted_value = \
                    calibration_curve(surrogateVerifications,surrogateForecasts,
                                      n_bins=nbins,normalize=False)
        #calibration_curve may return a smaller number of bins due to 
        #absence of values in certain intervals. Hence:
        true_nbins=len(fraction_of_positives) 
        #this loop is highly inefficient, I know
        for i in range(nbins):
            if i in range(true_nbins):
                pos.append(fraction_of_positives[i])
                mean.append(mean_predicted_value[i])
            else:
                pos.append(np.nan)
                mean.append(np.nan)
 
    for i in range(nbins):
        x=mean[i::nbins]
        y=pos[i::nbins]
        # print('x',len(x),np.nanmin(x),np.nanmax(x))
        # print('y',len(y),np.nanmin(y),np.nanmax(y))
        ytuples.append((np.nanpercentile(y,5),np.nanpercentile(y,95)))
        xtuples.append((np.nanmean(x),np.nanmean(x)))

    for x,y in zip(xtuples,ytuples):
       ax.plot(x,y,linestyle='--',color=color)
    
