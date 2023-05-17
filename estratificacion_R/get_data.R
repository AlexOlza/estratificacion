#!/usr/bin/Rscript
# (La linea de arriba es necesaria)
source('/home/aolza/Desktop/estratificacion/estratificacion_R/configuracion.R')
source('/home/aolza/Desktop/estratificacion/estratificacion_R/crear_tablas_CCS.R')
source('/home/aolza/Desktop/estratificacion/estratificacion_R/crear_tablas_ATC.R')

get_data <- function(year, binarize=TRUE){
  ccs <- get_CCS(year, binarize)
  atc <- get_ATC(year, binarize)
  X   <- merge(ccs,atc,on='PATIENT_ID')
  return(X)
}
if (sys.nframe() == 0){
X<- get_data(2016)
cost <- fread(as.character(config$ficheros_ACG['2016']),select=c('PATIENT_ID','COSTE_TOTAL_ANO2'))
descrip<-fread(config$fichero_descripciones_ccs)

##################################################################################################
###                 DESCRIPCIÓN DE LOS DATOS
##################################################################################################

dt <- data.table()
dt$CATEGORIES<-names(X[,-'PATIENT_ID'])
dt$N<-colSums(X[,-'PATIENT_ID'])
dt$percentage <- 100*dt$N/nrow(X)
dt <- merge(dt, descrip, on='CATEGORIES', all.x = TRUE)

# Pacientes con diabetes:
dt[dt$LABELS %like% '[Dd]iabetes',]
# Pacientes con hipertensión
dt[dt$LABELS %like% '[Hh]ypertension',]
##################################################################################################
###                 MODELO DE REGRESIÓN LINEAL
##################################################################################################
# La memoria no da para el modelo completo, lo hago con una muestra de los datos:
Xy <- merge(X,cost, on='PATIENT_ID')
sampleXy <-sample_n(Xy,100000)
model <- lm(COSTE_TOTAL_ANO2~., sampleXy[,-'PATIENT_ID'])
# Los coeficientes NO SON VÁLIDOS, pero se podrían describir así
coefs<-data.table(model$coefficients)
setnames(coefs,'beta')
coefs$CATEGORIES <- names(model$coefficients)
coefs <- as.data.table(merge(coefs, descrip, on='CATEGORIES', all.x = TRUE))
print(coefs[order(-beta)])
print(coefs[coefs$CATEGORIES=='FEMALE'])
print(coefs[startsWith(coefs$CATEGORIES,'AGE')])
}