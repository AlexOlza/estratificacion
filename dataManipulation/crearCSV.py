#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:27:47 2021

@author: aolza
"""
import config
import csv
import numpy as np
import re
import time
path=config.INDISPENSABLEDATAPATH+'modelMarkersActivos/'
outPath=config.DATAPATH+'modelMarkersActivos/'
parte1=['mm201612_1_activos.csv','mm201712_1_activos.csv','mm201812_1_activos.csv']
parte2=['mm201612_2_activos.csv','mm201712_2_activos.csv','mm201812_2_activos.csv']
outputFile=['2016-2017Activos.csv','2017-2018Activos.csv','2018-2019Activos.csv']
#PRIMERO CREAMOS LA CABECERA
columnas=''
variables,variablesEspeciales,long=[],[],[]
contador=[]
variables.append('PATIENT_ID')
with open("MODELMARKERS CARGA POSICIONES.txt") as f:
        read=csv.reader(f)
        next(read)
        next(read)
        for row in read:
            variables.append(re.sub('s2_acg_rec.','',row[0].split(' ')[0]).upper())
            if 'i:= i +' in str(row):
                # print(str(row))
                #Lleno el contador con las posiciones que hay que avanzar tras leer cada variable
                #Como no sé si el número será ded 2 o 1 cifras, necesito un try
                try:
                        contador.append(int(row[-1][-3:-1]))  #Si el número es de dos cifras
                except ValueError:
                        contador.append(int(row[-1][-2]))     #Si es de una

for f1,f2,out in zip(parte1,parte2,outputFile):
    print(f1)
    print(f2)
    print(out)
    f1=path+f1
    f2=path+f2
    out=outPath+out
    np.savetxt(out,[],delimiter=";",fmt="%s",header=';'.join(variables))
    t0=time.time()
    for f in [f1,f2]:
        with open(f) as prueba:
            read=csv.reader(prueba)
            #Estos archivos no tienen cabecera
            # if f==f1:
            #   next(read)
            with open(out,'a') as output:
                writer = csv.writer(output)
                for row in read:
                    row=row[0]
                    idMarkers=row.split("\t")
                    patientID=idMarkers[0]
                    modelMarkers=idMarkers[1]
                    m=list(modelMarkers)
                    j=0
                    M=[]
                    for i in range(len(contador)):
                        M.append(m[j])
                        j=j + contador[i]
                        # print(i,contador[i],j)
        
                    M=';'.join(M)
                    r=patientID+";"+M
                    writer.writerow([r])
        
    print('Tiempo de procesado: ',time.time()-t0 )