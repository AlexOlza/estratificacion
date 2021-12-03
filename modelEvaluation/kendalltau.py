#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONCORDANCIA DE LAS PREDICCIONES DE COSTE
Created on Tue Jul 20 13:19:37 2021

@author: aolza
"""

import pandas as pd
from scipy.stats import kendalltau
yr=18
linFile="linealPython/predictions/pred2018AgeSexDiagRxActivos.csv"
lassoFile="linealPython/predictions/lasso2018AgeSexDiagRxActivos.csv"
regrfFile="../cluster/coste/rfund/predictions/rfund302018AgeSexDiagRxActivos.csv"
reghgbFile="linealPython/predictions/hgb2018AgeSexDiagRxActivos.csv"
gammaFile="gamma/predictions/gamma2018AgeSexDiagRxActivos.csv"
tweedieFile="tweedie/predictions/tweedie1.2PredAgeSexDiagRx20{0}.csv".format(yr)
hgbFile="clasificacion/hgbsmallparamspace/probtop1_20{0}.csv".format(yr)
rfFile="clasificacion/randomForest/predictions/probtop1_20{0}recall.csv".format(yr)
logFile="logistica/logTop1PredAgeSexDiagRx20{0}Activos.csv".format(yr)
# nbFile="../../cluster/ingresos/adaBoostTree/predictions/urgSinPrevio18.csv"


hgb=pd.read_csv(hgbFile)
log=pd.read_csv(logFile)
rf=pd.read_csv(rfFile)
lin=pd.read_csv(linFile)
lasso=pd.read_csv(lassoFile)
regrf=pd.read_csv(regrfFile)
reghgb=pd.read_csv(reghgbFile)
tweedie=pd.read_csv(tweedieFile)
gamma=pd.read_csv(gammaFile)
# tweedie=pd.read_csv(tweedieFile)
# nb=pd.read_csv(nbFile)

reg=[lin,lasso,regrf,reghgb,tweedie,gamma]
alg=[log,hgb,rf]
regnames=['lin','lasso','regrf','reghgb','tweedie','gamma']
names=['log','hgb','rf']

for a in reg:
    print(a.columns)
    a['COSTE']=a[a.columns[1]]
    

#%%
used=[]
for alg1,n1 in zip(reg+alg,regnames+names):
    for alg2,n2 in zip(reg+alg,regnames+names):
        if n1==n2:
            continue
        if n1 not in ['gamma','tweedie']:
            continue
        if (n1+n2 in used) or (n2+n1 in used):
            continue
        used.append(n1+n2)
        print('tau ',n1,n2,kendalltau(alg1[alg1.columns[1]], alg2[alg2.columns[1]]))
        try:
            s1=alg1.sort_values(by='PROB_TOP1',ascending=False)
        except KeyError:
            s1=alg1.sort_values(by='COSTE',ascending=False)
            
        try:
            s2=alg2.sort_values(by='PROB_TOP1',ascending=False)
        except KeyError:
            s2=alg2.sort_values(by='COSTE',ascending=False)
        

        # l1=s1.PATIENT_ID[:100000]
        # l2=s2.PATIENT_ID[:100000]
        l1=s1.PATIENT_ID[:22405]
        l2=s2.PATIENT_ID[:22405]
        print('N=100000 intersection ',n1, 'and' , n2, len(list(set(l1).intersection(set(l2)))))