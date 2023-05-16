#!/usr/bin/Rscript
# (La linea de arriba es necesaria)
##################################################################################################################
#
#       LA FUNCIÓN create_ATC_table(year, binarize):
#         Devuelve la tabla de ATC del año correspondiente para los pacientes con algún fármaco.
#  ~~~~~~~~  MUY IMPORTANTE!!! Esta matriz sólo contiene a los pacientes con algún fármaco..
#  ~~~~~~~~                    Para conseguir a todos los pacientes con edad y sexo, usar get_X(year, binarize) !!!!
#         Si la encuentra en disco (en la direción que marque config$ficheros_atc[year]) la carga directamente,
#         si no, la calcula (tarda aprox. 45 minutos) y la guarda allí.
#         PASOS PARA CALCULARLA: Similares a los de create_CCS_table (leer esa documentación)
#########################################################################################################################
source('/home/aolza/Desktop/estratificacion/estratificacion_R/configuracion.R')
library(data.table)
library(stringr)
library(tictoc)
library(dplyr)

remove_non_alphanumeric <- function(string) return(str_replace_all(pattern = '[^[:alnum:]]', replacement='',string=string))

create_ATC_table <- function(year, binarize=TRUE){
  year <- as.character(year)
  if (file.exists(as.character(config$ficheros_atc[year]))){
    print(sprintf('Loading %s',config$ficheros_atc[year]))
    X <-fread(config$ficheros_atc[year],colClasses = 'integer')
    if (binarize){ X[, names(X)[-1] := lapply(.SD, function(x) as.integer(x!=0)), .SDcols = 2:ncol(X)]}
    return(X)
    
  }
  print('Creating ATC table. This may take a while (between half an hour and an hour).')
  # Quality checks, and verification that the needed files exist
  if ( is.na(as.integer(year))) stop('year must be an integer number, or a string that represents an integer!')
  if ( !(file.exists(as.character(config$ficheros_x[year])))) stop(sprintf('Missing file: config$ficheros_rx[year]. %s does not exist.',config$ficheros_rx[year]))
  
  # Reading dictionary and prescriptions
  atc_dict <- fread(config$diccionario_atc)
  rx <- fread(as.character(config$ficheros_rx[year]))
  setnames(rx, c('PATIENT_ID','date','CODE','a','number' ))
  
  # PREPROCESSING 
  atc_dict <- atc_dict[, lapply(.SD, remove_non_alphanumeric)]
  atc_dict <- atc_dict[, lapply(.SD, toupper)]
  rx$CODE <- remove_non_alphanumeric(rx$CODE)
  rx <- rx[, lapply(.SD, toupper),.SDcols=c('CODE')]
  
  unique_codes_prescribed<-data.frame('CODE'=unique(rx$CODE))
  
  
  for (start in unique(atc_dict$starts_with)) {
    unique_codes_prescribed[startsWith(as.character(unique_codes_prescribed$CODE),start),'drug_group']=atc_dict[atc_dict$starts_with==start, 'drug_group'][1]
  }
  n_distinct_drugs <- length(unique(unique_codes_prescribed$drug_group))
  print(sprintf('In year %s, %s distinct drug groups from the dictionary were prescribed to patients',year,n_distinct_drugs))
  
  #Drop codes that were not prescribed to any patient in the current year
  rx_with_drug_group=pd.DataFrame({'PATIENT_ID':[],'CODE':[],'drug_group':[]})
  # df=diags.copy()
  rx_with_drug_group=pd.merge(rx, unique_codes_prescribed, on=['CODE'], how='inner')[['PATIENT_ID','CODE','drug_group']].dropna()
  ATC_drug_group_descriptive(rx_with_drug_group,yr)
  
  # COMPUTE THE DATA MATRIX (takes half an hour)
  tic('Computing ATC matrix')
  X=data.frame('PATIENT_ID'=unique(diags_with_ccs$PATIENT_ID))
  i=0
  grouped <-diags_with_ccs %>% group_by(CCS,PATIENT_ID)
  amount_per_patient<- grouped %>% summarize(count=n())
  I<-length(unique(amount_per_patient$CCS))
  for (ccs_number in unique(amount_per_patient$CCS)){
    i<-i+1
    patients_with_CCS <- amount_per_patient[amount_per_patient$CCS==ccs_number,c('PATIENT_ID','count')]
    print(sprintf('CCS %s (%s percent done)',ccs_number,i*100/I))
    X=merge(X,patients_with_CCS,all.x = TRUE)
    names(X)[names(X) == "count"] <- paste('CCS',ccs_number,sep='')
  }
  X[is.na(X)] <- 0
  toc()
  fwrite(X,as.character(config$ficheros_ccs[year]))
  if (binarize){X[, names(X)[-1] := lapply(.SD, function(x) as.integer(x!=0)), .SDcols = 2:ncol(X)]}
  return(X)
}

get_CCS <- function(year, binarize=TRUE){
  ##########################################################################################
  #     Returns the CCS table for all patients in the Basque Country (including those without any illness)
  ###########################################################################################
  ccs <- create_CCS_table(year, binarize)
  # I read the first row to extract the column names
  column_names <-  names(fread(as.character(config$ficheros_ACG[as.character(year)]),nrows = 1))
  # I now read only columns for ID, sex and age, EXCEPT AGE_85GT (that is a linear combination of the others)
  all_patients <- fread(as.character(config$ficheros_ACG[as.character(year)]),
                        select=column_names[! is.na(str_match(column_names,'PATIENT_ID|FEMALE|AGE_[0-9]+$'))])
  X <- ccs[all_patients, on = .(PATIENT_ID)]
  X <-setnafill(X, fill = 0)
  predictors <- names(X)[names(X)!='PATIENT_ID']
  setcolorder(X, c('PATIENT_ID',str_sort(predictors))) 
  # I return X with the columns in alphabetical order. This is important for modelling (they must be always in the same order)
  return(X)
}

year <-2017
ccs <- get_CCS(year)
