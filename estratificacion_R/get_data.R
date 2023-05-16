source('/home/aolza/Desktop/estratificacion/estratificacion_R/configuracion.R')
source('/home/aolza/Desktop/estratificacion/estratificacion_R/crear_tablas_CCS.R')
source('/home/aolza/Desktop/estratificacion/estratificacion_R/crear_tablas_ATC.R')

get_data <- function(year, binarize=TRUE){
  ccs <- get_CCS(year, binarize)
  atc <- get_ATC(year, binarize)
  X   <- merge(ccs,atc,on='PATIENT_ID')
  return(X)
}

X<- get_data(2016)
cost <- fread(as.character(config$ficheros_ACG['2016']),select=c('PATIENT_ID','COSTE_TOTAL_ANO2'))

Xy <- merge(X,cost, on='PATIENT_ID')
sampleXy <-sample_n(Xy,10000)
model <- lm(COSTE_TOTAL_ANO2~., sampleXy[,-'PATIENT_ID'])

print(sort(model$coefficients, decreasing=TRUE))
