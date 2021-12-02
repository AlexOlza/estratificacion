#SCRIPT PARA ORGANIZAR LAS BASES DE DATOS
#-Cargar la base de datos original
#-Filtrar quedándose sólo con los pacientes en SITUACION_ANO2==D (difunto) o A (activo)
#-Cargar la BD de costes y OSI
#-Comprobar que los ID de pacientes coinciden
#-Corregir los costes
#-Añadir la OSI
#-Guardar el dataframe resultante
library(dplyr)
library(data.table)
mainDir<-getwd()


#       VARIABLES DE 2016 CON LOS COSTES DE 2017                                         #
##########################################################################################
ano12017<-fread("estratificacionDatos/obsoleto/2017Ano1.txt",sep=";")
subDir<-"estratificacionDatos/2016"
setwd(file.path(mainDir, subDir))
summary(ano12017$SITUACION_ANO2) 
#- Filtro difunto o activo es equivalente a quedarse solo con
#los ID que estén en costesOSI
nrow(ano12017[c(ano12017$SITUACION_ANO2=="A",ano12017$SITUACION_ANO2=="D"),])
# Vemos que hay 2240526 personas con D o A. Es lo que esperábamos.
#- Cargo BD de costes y OSI
costesOSI<-fread("pob2016_costes2017.txt",sep="\t")
ano12017<-subset(ano12017,PATIENT_ID%in%costesOSI$id_paciente)
summary(ano12017$SITUACION_ANO2)
head(costesOSI) #se ha leido bien
#- Añadimos los costes y la OSI

costesOSI$COSTE_TOTAL_ANO2<-costesOSI$costetotal
costesOSI$PATIENT_ID<-costesOSI$id_paciente
costesOSI<-subset(costesOSI,select=-c(id_paciente,costetotal))
head(costesOSI)
ano12017<-subset(ano12017,select = -COSTE_TOTAL_ANO2)
a2016_2017<-inner_join(ano12017,costesOSI,by="PATIENT_ID")
summary(a2016_2017)#No hay NA inesperados

#ELIMINAMOS LAS VARIABLES COSTE_FARMACIA_ANO2 Y SITUACION_ANO2 porque no están en todos los datasets
#y necesitamos que todos tengan la misma estructura
a2016_2017<-subset(a2016_2017,select=-c(SITUACION_ANO2,COSTE_FARMACIA_ANO2))

write.csv(a2016_2017,"../2016-2017.txt",sep=',')
# a2016_2017<-read.csv("../2016-2017.txt")
# COLINEALIDAD
summary(a2016_2017$AGE_0004+a2016_2017$AGE_0511+a2016_2017$AGE_1217+a2016_2017$AGE_1834+a2016_2017$AGE_3544+
          a2016_2017$AGE_4554+a2016_2017$AGE_5564+a2016_2017$AGE_6569+a2016_2017$AGE_7074+a2016_2017$AGE_7579+
          a2016_2017$AGE_8084+a2016_2017$AGE_85GT) #SON MUTUAMENTE EXCLUYENTES
summary(a2016_2017$PRORUB_3-a2016_2017$ACGRUB_3) #SON IGUALES
rm(ano12017,costesOSI,a2016_2017)

#       VARIABLES DE 2017 CON LOS COSTES DE 2018                                         #
##########################################################################################
subDir<-"estratificacionDatos/2017"
setwd(file.path(mainDir, subDir))
ano22017<-fread("2017Ano2.txt",sep=";") #2256287 obs
#Numero de difuntos y activos no disponible
#No hay variables de coste año 2, hay que llenarlas
#- Cargo BD de costes y OSI
costesOSI<-fread("pob2017_costes2018.txt",sep="\t")
#- FILTRADO: Nos quedamos con los ID que estén en costesOSI
ano22017<-subset(ano22017,PATIENT_ID%in%costesOSI$id_paciente)

#- AÑADIDO DE COSTES Y OSI
costesOSI$COSTE_TOTAL_ANO2<-costesOSI$costetotal
costesOSI$PATIENT_ID<-costesOSI$id_paciente
costesOSI<-subset(costesOSI,select=-c(id_paciente,costetotal))
head(costesOSI)
summary(costesOSI$osi)
a2017_2018<-inner_join(ano22017,costesOSI,by="PATIENT_ID")
summary(a2017_2018$osi)

write.csv(a2017_2018,"../2017-2018.txt",sep=',')
a2017_2018<-read.csv("../2017-2018.txt",sep=',')
# COLINEALIDAD
summary(a2017_2018$AGE_0004+a2017_2018$AGE_0511+a2017_2018$AGE_1217+a2017_2018$AGE_1834+a2017_2018$AGE_3544+
          a2017_2018$AGE_4554+a2017_2018$AGE_5564+a2017_2018$AGE_6569+a2017_2018$AGE_7074+a2017_2018$AGE_7579+
          a2017_2018$AGE_8084+a2017_2018$AGE_85GT) #SON MUTUAMENTE EXCLUYENTES
summary(a2017_2018$PRORUB_3-a2017_2018$ACGRUB_3) #SON IGUALES
rm(ano22017,costesOSI,a2017_2018)
#       VARIABLES DE 2018 CON LOS COSTES DE 2019                                         #
##########################################################################################
subDir<-"estratificacionDatos/2018"
setwd(file.path(mainDir, subDir))
ano12018<-fread("201912_1_S2_Ano1.TXT",sep=";") 

#Hay que reemplazar COSTETOTAL ANO2, para ello eliminamos esa columna
ano12018<-subset(ano12018,select = -COSTE_TOTAL_ANO2)
#- Difuntos y activos, hay 2254652
nrow(ano12018[c(ano12018$SITUACION_ANO2=="A",ano12018$SITUACION_ANO2=="D"),])
summary(ano12018$COSTE_TOTAL_ANO2)

#- Cargo BD de costes y OSI
costesOSI<-fread("pob2018_costes2019.txt", header = TRUE, sep="\t")
ano12018<-subset(ano12018,PATIENT_ID%in%costesOSI$id_paciente) 

#- AÑADIDO DE COSTES Y OSI
costesOSI$COSTE_TOTAL_ANO2<-costesOSI$costetotal2019
costesOSI$PATIENT_ID<-costesOSI$id_paciente
costesOSI<-subset(costesOSI,select=-c(id_paciente,costetotal2019))
head(costesOSI)
summary(costesOSI$osi)
a2018_2019<-inner_join(ano12018,costesOSI,by="PATIENT_ID")
summary(a2018_2019$PATIENT_ID)

#ELIMINAMOS LAS VARIABLES COSTE_FARMACIA_ANO2 Y SITUACION_ANO2 porque no están en todos los datasets
#y necesitamos que todos tengan la misma estructura
a2018_2019<-subset(a2018_2019,select=-c(SITUACION_ANO2,COSTE_FARMACIA_ANO2))


write.csv(a2018_2019,"../2018-2019.txt",sep=',')

a2018_2019<-read.csv("../2018-2019.txt",sep=',')
# COLINEALIDAD
summary(a2018_2019$AGE_0004+a2018_2019$AGE_0511+a2018_2019$AGE_1217+a2018_2019$AGE_1834+a2018_2019$AGE_3544+
          a2018_2019$AGE_4554+a2018_2019$AGE_5564+a2018_2019$AGE_6569+a2018_2019$AGE_7074+a2018_2019$AGE_7579+
          a2018_2019$AGE_8084+a2018_2019$AGE_85GT) #SON MUTUAMENTE EXCLUYENTES
summary(a2018_2019$PRORUB_3-a2018_2019$ACGRUB_3) #SON IGUALES

rm(ano12018,a2018_2019,costesOSI)
