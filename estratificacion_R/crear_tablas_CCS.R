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
  
  # PREPROCESSING ICD9 DICTIONARY
      icd9 <- icd9[, lapply(.SD, as.character)]  # convert all columns to character
      #remove non-alphanumeric characters only in columns that do not contain the name LABEL (because we need those to extract the CCS)
      non_label_cols <- names(icd9)[!grepl( 'LABEL', names(icd9), fixed = TRUE)]
      icd9[, non_label_cols] <- icd9[, lapply(.SD, remove_non_alphanumeric),.SDcols=non_label_cols]  
      icd9 <- icd9[, lapply(.SD, toupper)]  # everything to uppercase
      #The description can be in any of these columns, so we paste them
      icd9$DESCRIPTION <- paste(icd9$`CCS LVL 1 LABEL`,
                                 icd9$`CCS LVL 2 LABEL`,
                                 icd9$`CCS LVL 3 LABEL`,
                                 icd9$`CCS LVL 4 LABEL`,sep = ';')
      #extract the number that is followed by a dot and a bracket. This is the CCS number.
      #Example: "... BACTERIAL INFECTION;TUBERCULOSIS [1.]" ----> "1"
      icd9[, CCS := (str_extract(DESCRIPTION, '([0-9]+)(?=\\.\\])'))]  
      # Truncate the code at the 5th digit, add the ICD version (which is 9), rename columns
      colnames(icd9)[colnames(icd9) == 'ICD-9-CM CODE']<-'CODE'
      icd9$CODE<-substr(icd9$CODE, 1, 5)
      icd9$CIE_VERSION<-'9'
      icd9<-icd9[, c('CODE','CCS','DESCRIPTION','CIE_VERSION')]
      
  # PREPROCESSING ICD10CM DICTIONARY
      colnames(icd10cm)[colnames(icd10cm) == 'ICD-10-CM CODE']<-'CODE'
      colnames(icd10cm)[colnames(icd10cm) == 'CCS CATEGORY']<-'CCS'
      colnames(icd10cm)[colnames(icd10cm) == 'CCS CATEGORY DESCRIPTION']<-'DESCRIPTION'
      icd10cm <- icd10cm[, lapply(.SD, as.character)]  # convert all columns to character
      #remove non-alphanumeric characters only in columns that we will be using
      icd10cm[, c('CODE','CCS')] <- icd10cm[, lapply(.SD, remove_non_alphanumeric),.SDcols=c('CODE','CCS')]  
      icd10cm <- icd10cm[, lapply(.SD, toupper)]  # everything to uppercase
      icd10cm$CIE_VERSION<-'10'
      icd10cm<-icd10cm[, c('CODE','CCS','DESCRIPTION','CIE_VERSION')]
      
  # PREPROCESSING DIAGNOSES
      colnames(diags)[colnames(diags) == 'CIE_CODE'] <-'CODE'
      diags[, c('CODE','CIE_VERSION')] <- diags[, lapply(.SD, as.character),.SDcols=c('CODE','CIE_VERSION')]
      diags[, c('CODE','CIE_VERSION')] <- diags[, lapply(.SD, remove_non_alphanumeric),.SDcols=c('CODE','CIE_VERSION')] 
      
      diags[startsWith(diags$CIE_VERSION,'9'),'CIE_VERSION']='9'
      diags[startsWith(diags$CIE_VERSION,'10'),'CIE_VERSION']='10'

      diags[diags$CIE_VERSION=='10' & grepl("^[0-9]", diags$CODE),'CODE']='ONCOLOGY'
}

year <-2016
#ccs <- CCS_table(year)
