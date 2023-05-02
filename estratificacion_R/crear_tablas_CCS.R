source('/home/aolza/Desktop/estratificacion/estratificacion_R/configuracion.R')
library(data.table)
library(stringr)

remove_non_alphanumeric <- function(string) return(str_replace_all(pattern = '[^[:alnum:]]', replacement='',string=string))

CCS_table <- function(year){
  year <- as.character(year)
  if (file.exists(as.character(config$ficheros_ccs[year]))){
    print(sprintf('Loading %s',config$ficheros_ccs[year]))
    #ccs <-fread(config$ficheros_ccs[year],colClasses = 'integer')
    #return(ccs)
  
  }
  print('Creating CCS table. This may take a while.')
  # Quality checks, and verification that the needed files exist
  if ( is.na(as.integer(year))) stop('year must be an integer number, or a string that represents an integer!')
  if ( !(file.exists(as.character(config$diccionario_cie9_ccs)))) stop(sprintf('Missing file: config$diccionario_cie9_ccs. %s does not exist.',config$diccionario_cie9_ccs))
  if ( !(file.exists(as.character(config$diccionario_cie10cm_ccs)))) stop(sprintf('Missing file: config$diccionario_cie10cm_ccs. %s does not exist.',config$diccionario_cie10cm_ccs))
  if ( !(file.exists(as.character(config$ficheros_dx[year])))) stop(sprintf('Missing file: config$ficheros_dx[year]. %s does not exist.',config$ficheros_dx[year]))
  
  # Reading dictionaries and diagnoses
  icd9    <- fread(as.character(config$diccionario_cie9_ccs))
  icd10cm <- fread(as.character(config$diccionario_cie10cm_ccs))
  diags   <- fread(as.character(config$ficheros_dx[year]),select=c('PATIENT_ID','CIE_VERSION','CIE_CODE'))
  
  # Preprocessing
  
  apply(icd9,2,as.character)
  apply(icd9,2,sub(pattern = '[^a-zA-Z\d]', replacement=''))
  icd9 <- icd9[, lapply(.SD, as.character)]  # convert all columns to character
  #icd9 <- icd9[, lapply(.SD, remove_non_alphanumeric)]  #remove non-alphanumeric characters ONLY CERTAIN COLS
  icd9 <- icd9[, lapply(.SD, toupper)]  # everything to uppercase

 
}

year <-2016
#ccs <- CCS_table(year)
