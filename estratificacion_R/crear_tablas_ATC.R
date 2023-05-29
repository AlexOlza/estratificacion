#!/usr/bin/Rscript
# (La linea de arriba es necesaria)
##################################################################################################################
#
#       LA FUNCIÓN create_ATC_table(year, binarize):
#         Devuelve la tabla de ATC del año correspondiente para los pacientes con algún fármaco.
#  ~~~~~~~~  MUY IMPORTANTE!!! Esta matriz sólo contiene a los pacientes con algún fármaco..
#  ~~~~~~~~                    Para conseguir a todos los pacientes con edad y sexo, usar get_data(year, binarize) !!!!
#         Si la encuentra en disco (en la direción que marque config$ficheros_atc[year]) la carga directamente,
#         si no, la calcula (tarda aprox. 45 minutos) y la guarda allí.
#         PASOS PARA CALCULARLA: Similares a los de create_CCS_table (leer esa documentación)
#########################################################################################################################
source('/home/alex/Desktop/estratificacion/estratificacion_R/configuracion.R')
library(data.table)
library(stringr)
library(tictoc)
library(dplyr)

remove_non_alphanumeric <- function(string) return(str_replace_all(pattern = '[^[:alnum:]]', replacement='',string=string))

ATC_drug_group_descriptive <- function(rx_with_drug_group, year){
  number_of_cases=list()
  grouped <-rx_with_drug_group %>% group_by('CODE')
  for (code in unique(grouped$CODE)){
    df <- grouped[grouped$CODE==code,]
    print(code)
    number_of_cases[code]<-length(unique(df$PATIENT_ID))
  }
  unique_rx_with_group<-unique(rx_with_drug_group,by=c('CODE','drug_group'))[,c('CODE','drug_group')]
  cases <- data.table()
  cases$N <- transpose(data.table(do.call(cbind, number_of_cases)))
  cases$CODE <- names(number_of_cases)
  df <- merge(unique_rx_with_group, cases, by='CODE')
  number_of_cases_per_drug_group <- aggregate(N ~ drug_group, data=df, FUN=sum)
  setorder(number_of_cases_per_drug_group, cols = - "N") 
  fwrite(number_of_cases_per_drug_group,
         file.path(config$carpeta_ficheros_auxiliares,paste('number_of_cases_per_drug_group_',year,'.xlsx',sep='')))
}
  

create_ATC_table <- function(year, binarize=TRUE){
  year <- as.character(year)
  if (file.exists(as.character(config$ficheros_atc[year]))){
    print(sprintf('Loading %s',config$ficheros_atc[year]))
    X <-fread(config$ficheros_atc[year],colClasses = 'integer')
    if (binarize){ X[, names(X)[-1] := lapply(.SD, function(x) as.integer(x!=0)), .SDcols = 2:ncol(X)]}
    return(X)
    
  }
  print('Creating ATC table. This may take a while (between 10min and half an hour).')
  # Quality checks, and verification that the needed files exist
  if ( is.na(as.integer(year))) stop('year must be an integer number, or a string that represents an integer!')
  if ( !(file.exists(as.character(config$ficheros_rx[year])))) stop(sprintf('Missing file: config$ficheros_rx[year]. %s does not exist.',config$ficheros_rx[year]))
  
  # Reading dictionary and prescriptions
  atc_dict <- fread(config$diccionario_atc)
  rx <- fread(as.character(config$ficheros_rx[year]))
  setnames(rx, c('PATIENT_ID','date','CODE','a','number' ))
  
  # PREPROCESSING 
  atc_dict <- atc_dict[, lapply(.SD, remove_non_alphanumeric)]
  atc_dict <- atc_dict[, lapply(.SD, toupper)]
  rx[, CODE := lapply(.SD, remove_non_alphanumeric),.SDcols=c('CODE')]
  rx[, CODE := lapply(.SD, toupper),.SDcols=c('CODE')]
  
  unique_codes_prescribed<-data.table('CODE'=unique(rx$CODE))
  
  
  for (start in unique(atc_dict$starts_with)) {
    unique_codes_prescribed[startsWith(as.character(unique_codes_prescribed$CODE),start),'drug_group']=atc_dict[atc_dict$starts_with==start, 'drug_group'][1]
  }
  n_distinct_drugs <- length(unique(unique_codes_prescribed$drug_group))
  print(sprintf('In year %s, %s distinct drug groups from the dictionary were prescribed to patients',year,n_distinct_drugs))
  
  rx_with_drug_group<-rx[ unique_codes_prescribed, on='CODE']
  rx_with_drug_group <- rx_with_drug_group[!is.na(rx_with_drug_group$drug_group)]
  ATC_drug_group_descriptive(rx_with_drug_group,year)
  
  # COMPUTE THE DATA MATRIX (takes 10min)
  tic('Computing ATC matrix')
  X=data.table('PATIENT_ID'=unique(rx_with_drug_group$PATIENT_ID))
  i=0
  grouped <-rx_with_drug_group %>% group_by(drug_group,PATIENT_ID)
  amount_per_patient<- grouped %>% summarize(count=n())
  I<-length(unique(amount_per_patient$drug_group))
  for (atc_group in unique(amount_per_patient$drug_group)){
    i<-i+1
    patients_with_group <- amount_per_patient[amount_per_patient$drug_group==atc_group,c('PATIENT_ID','count')]
    print(sprintf('ATC %s (%s percent done)',atc_group,i*100/I))
    X=merge(X,patients_with_group,all.x = TRUE)
    names(X)[names(X) == "count"] <- paste('PHARMA_',atc_group,sep='')
  }
  X[is.na(X)] <- 0
  toc()
  fwrite(X,as.character(config$ficheros_atc[year]))
  if (binarize){X[, names(X)[-1] := lapply(.SD, function(x) as.integer(x!=0)), .SDcols = 2:ncol(X)]}
  return(X)
}

get_ATC <- function(year, binarize=TRUE){
  ##########################################################################################
  #     Returns the CCS table for all patients in the Basque Country (including those without any illness)
  ###########################################################################################
  atc <- create_ATC_table(year, binarize)
  # I read only column for ID and gender
  all_patients <- fread(as.character(config$ficheros_ACG[as.character(year)]),select=c('PATIENT_ID','FEMALE'))
  X <- atc[all_patients, on = .(PATIENT_ID)]
  X <-setnafill(X, fill = 0)
  # Condition 1: PHARMA_BENIGNPROSTATICHYPERPLASIA must be zero for females
  X[X$FEMALE==1,"PHARMA_BENIGNPROSTATICHYPERPLASIA"] <- 0
  # Condition 2: Congestive_heart_failure must have at least one in every block
  X$PHARMA_CONGESTIVEHEARTFAILURE <- 0
  congestive <- c("PHARMA_CONGESTIVEHEARTFAILUREBLOCK1",
                  "PHARMA_CONGESTIVEHEARTFAILUREBLOCK2",
                  "PHARMA_CONGESTIVEHEARTFAILUREBLOCK3")
  X[rowSums(X[,..congestive])==3,"PHARMA_CONGESTIVEHEARTFAILURE"] <- 1
  X <- X[,(congestive):=NULL]
  predictors <- names(X)[names(X)!='PATIENT_ID']
  setcolorder(X, c('PATIENT_ID',str_sort(predictors))) 
  # I return X with the columns in alphabetical order. This is important for modelling (they must be always in the same order)
  return(X)
}
# The statements below are not executed if this script is sourced
if (sys.nframe() == 0){
year <-2017
atc <- get_ATC(year)
}