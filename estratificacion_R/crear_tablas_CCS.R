source('/home/aolza/Desktop/estratificacion/estratificacion_R/configuracion.R')
library(data.table)
library(stringr)

remove_non_alphanumeric <- function(string) return(str_replace_all(pattern = '[^[:alnum:]]', replacement='',string=string))

#Functions that do the same as I did in Python
missingDX <- function(dic,diag) return(unique(diag$CODE[!diag$CODE %in% unique(dic$CODE)])) #detects missing diagnoses

guessingCCS <- function(missingdx, dictionary){  # asume el CCS quitando "lo que cuelga"
  success <- list() ; failure <- list()
  for (dx in missingdx) {
    if (! dx=='ONCOLO'){
      i<-1
      options<-unique(dictionary[startsWith(dictionary$CODE,dx)]$CCS)
      code<-dx
      while ((length(options)==0) & (i<=length(dx))){ 
        i=i+1
        code <- str_sub(dx,end=-i)
        options<-unique(dictionary[startsWith(dictionary$CODE,code)]$CCS)
      }
      if (length(options)>=2) options <- options[options != '259'] #CCS 259 is for residual unclassified codes
      if (length(options)==1) {success[dx] <- options} 
      else {
        failure <- c(failure,dx)}
    }
  }
  return(list('success'=success,'failure'=failure))
}

needsManualRevision <- function(failure, dictionary,filename){
  filename <- paste(config$carpeta_datos_indispensable,'/ccs/manually_revised_icd9.csv',sep='')
  if (file.exists(filename)){
    already_there <- fread(filename)
    append_to_csv <- TRUE
  }else  {append_to_csv <- FALSE ; already_there <- data.frame()}
 
 
}

keys_to_revise=[k[0] for k in failure.keys()]
with open(f'needs_manual_revision{appendix}.csv', mode) as output:
  writer = csv.writer(output)
if not all([k in already_there.CODE.values for k in keys_to_revise]):
  if mode=='w': writer.writerow(['CODE','N'])
for key in keys_to_revise:
  if not key in already_there.CODE.values: 
  N=len(diags.loc[diags.CODE==key].PATIENT_ID.unique())
writer.writerow([key, N]) 

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
      #extract the number that is followed by a dot (optional) and a bracket. This is the CCS number.
      #Example: "... BACTERIAL INFECTION;TUBERCULOSIS [1.]" ----> "1"
      icd9[, CCS := (str_extract(DESCRIPTION, '([0-9]+)(?=\\.?\\])'))]  
      # Truncate the code at the 5th digit, add the ICD version (which is 9), rename columns
      colnames(icd9)[colnames(icd9) == 'ICD-9-CM CODE']<-'CODE'
      icd9$CODE<-substr(icd9$CODE, 1, 5)
      icd9$CIE_VERSION<-'9'
      icd9<-icd9[, c('CODE','CCS','CIE_VERSION')]
      
  # PREPROCESSING ICD10CM DICTIONARY
      colnames(icd10cm)[colnames(icd10cm) == 'ICD-10-CM CODE']<-'CODE'
      colnames(icd10cm)[colnames(icd10cm) == 'CCS CATEGORY']<-'CCS'
      colnames(icd10cm)[colnames(icd10cm) == 'CCS CATEGORY DESCRIPTION']<-'DESCRIPTION'
      icd10cm <- icd10cm[, lapply(.SD, as.character)]  # convert all columns to character
      #remove non-alphanumeric characters only in columns that we will be using
      icd10cm[, c('CODE','CCS')] <- icd10cm[, lapply(.SD, remove_non_alphanumeric),.SDcols=c('CODE','CCS')]  
      icd10cm <- icd10cm[, lapply(.SD, toupper)]  # everything to uppercase
      icd10cm$CIE_VERSION<-'10'
      icd10cm<-icd10cm[, c('CODE','CCS','CIE_VERSION')]
      
  # PREPROCESSING DIAGNOSES
      colnames(diags)[colnames(diags) == 'CIE_CODE'] <-'CODE' #for uniformity, I rename the column
      diags[, c('CODE','CIE_VERSION')] <- diags[, lapply(.SD, as.character),.SDcols=c('CODE','CIE_VERSION')] #every column as character
      diags[, c('CODE','CIE_VERSION')] <- diags[, lapply(.SD, remove_non_alphanumeric),.SDcols=c('CODE','CIE_VERSION')] #remove non-alphanumeric chars including spaces
      
      diags[startsWith(diags$CIE_VERSION,'9'),'CIE_VERSION']='9'
      diags[startsWith(diags$CIE_VERSION,'10'),'CIE_VERSION']='10'
      
      # ICD10CM diagnoses that begin with a number are grouped all together
      diags[diags$CIE_VERSION=='10' & grepl("^[0-9]", diags$CODE),'CODE']='ONCOLOGY'
      
      # We truncate ICD9 codes and ICD10CM codes at the 5th and the 6th position, respectively
      # we act only on the necessary lines, otherwise this is very slow
      diags[diags$CIE_VERSION=='9' & nchar(diags$CODE)>5, CODE := str_sub(diags[diags$CIE_VERSION=='9' & nchar(diags$CODE)>5, 'CODE'], 1,5)]
      diags[diags$CIE_VERSION=='10' & nchar(diags$CODE)>6, CODE := str_sub(diags[diags$CIE_VERSION=='10' & nchar(diags$CODE)>6,'CODE'], 1,6)]
      
      # We drop null diagnoses
      print(sprintf('Dropping %s NULL codes:',nrow(diags[is.na(diags$CODE),])))
      diags<-na.omit(diags,cols='CODE')

  # QUALITY CHECKS: Check that all the dx in the icd9 and icd10cm dictionaries have an assigned CCS
      if (any(is.na(icd9$CCS))) stop('Some codes in the ICD9 dictionary have not been assigned a CCS :(')
      if (any(is.na(icd10cm$CCS))) stop('Some codes in the ICD10CM dictionary have not been assigned a CCS :(')
      if (any(is.na(diags))) stop('Null values encountered after preprocessing the diagnoses :(')
      
  # PERFORM MANUAL REVISION ON MISSING CODES
      missing_in_icd9<-missingDX(icd9,diags[diags$CIE_VERSION=='9'])
      missing_in_icd10cm<-missingDX(icd10cm,diags[diags$CIE_VERSION=='10'])
      print(sprintf('Missing quantity ICD9: %s', length(missing_in_icd9)))
      print(sprintf('Missing quantity ICD10: %s', length(missing_in_icd10cm)))
      guesses9=guessingCCS(missing_in_icd9, icd9)
      sprintf('In ICD9, %s codes need manual revision',length(guesses9$failure))
      sprintf('In ICD9, %s codes have been assigned',length(guesses9$success))
      #icd9=assign_success(success9,icd9)
      
      guesses10=guessingCCS(missing_in_icd10cm, icd10cm)
      sprintf('In ICD10CM, %s codes need manual revision',length(guesses10$failure))
      sprintf('In ICD10CM, %s codes have been assigned',length(guesses10$success))
      
      # Add successful codes to ICD9 and ICD10CM dictionaries
      for(i in seq_along(guesses9$success)){
        new_row=cbind('CODE'=names(guesses9$success)[i], 'CCS'=guesses9$success[[i]],'CIE_VERSION'='9')
        icd9=rbind(icd9,new_row)
      }
      
      for(i in seq_along(guesses10$success)){
        new_row=cbind('CODE'=names(guesses10$success)[i], 'CCS'=guesses10$success[[i]],'CIE_VERSION'='10')
        icd10cm=rbind(icd10cm,new_row)
      }
      
     
      
}

year <-2016
#ccs <- CCS_table(year)
