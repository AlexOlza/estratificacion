

createYearlyDataFrames <- function(){
  # Devuelve una lista de data.frames: Un data.frame para cada año.
  # Cada data.frame contiene los ingresos de el año correspondiente
  ing <- fread(config$fichero_todos_los_ingresos)
  ing$fecing <- as.Date(ing$fecing) #fecing: fecha de ingreso
  years <- unique(year(ing$fecing)) #el fichero contiene ingresos con fechas en 2016, 17 y 18
  L=list()
  for (y in years){
    dt <- ing[year(fecing)==y,]
    dic <- transform_ing(dt)
    L[[y]] <- dic
  }
  return(L)
}
 

transform_ing <- function(dt){ 
  # Función pensada para ser llamada sólo desde createYearlyDataFrames. 
  # Devuelve un data.frame con los ingresos de cada paciente. Las columnas representan el tipo de ingresos,
  # y las filas el número de ingresos de cada tipo que tuvo ese paciente.
  needed=c('id', 'tipo','prioridad','planned_cms', 'newborn_injury')
  for (column in needed){if (!(column %in% names(dt))) stop(sprintf('Missing column in hospitalization file: %s', column))}
  if (!('ing' %in% names(dt)))
    dt$ing=1
  
  #Characteristics of each hospitalization (boolean arrays)
  urg=(dt$prioridad=='URGENTE')
  plancms=(dt$planned_cms==1)
  hdia=(dt$tipo=='h.dia')
  nbinj=(dt$newborn_injury==1)
  
  #auxiliary cols
  dt$nbinj=dt$newborn_injury  #shorter name
  
  #Describing all the characteristics of each episode
  #And computing the number of similar episodes per id using groupby and transform(sum)
  ###################################################
  df <- data.frame('id'=unique(dt$id))
  #PRIORITY ACCORDING TO OSAKIDETZA ADMINISTRATIVE CRITERIA
  dt$urg=as.integer(urg)
  dt$prog=as.integer(!urg)
  df <- merge((dt %>% group_by(id) %>% summarise(urg_num=sum(urg))), df, on=id,all.y=TRUE)
  df <- merge((dt %>% group_by(id) %>% summarise(prog_num=sum(prog))), df, on=id,all.y=TRUE)
  
  #PRIORITY ACCORDING TO THE CMS ALGORITHM
  dt$urgcms=as.integer(!plancms) 
  dt$progcms=as.integer(plancms)
  df <- merge((dt %>% group_by(id) %>% summarise(urgcms_num=sum(urgcms))), df, on=id,all.y=TRUE)
  df <- merge((dt %>% group_by(id) %>% summarise(progcms_num=sum(progcms))), df, on=id,all.y=TRUE)
  
  
  #POTENTIAL EXCLUSION CRITERIA: Hospitalizations due to birth/delivery/traumatic injury
  dt$nbinj_urg=as.integer((urg) & (nbinj))
  dt$nbinj_prog=as.integer((!urg) & (nbinj))
  dt$nbinj_urgcms=as.integer((!plancms) & (nbinj))
  dt$nbinj_progcms=as.integer((plancms) & (nbinj))
  df <- merge((dt %>% group_by(id) %>% summarise(nbinj_urg_num=sum(nbinj_urg))), df, on=id,all.y=TRUE)
  df <- merge((dt %>% group_by(id) %>% summarise(nbinj_prog_num=sum(nbinj_prog))), df, on=id,all.y=TRUE)
  df <- merge((dt %>% group_by(id) %>% summarise(nbinj_urgcms_num=sum(nbinj_urgcms))), df, on=id,all.y=TRUE)
  df <- merge((dt %>% group_by(id) %>% summarise(nbinj_progcms_num=sum(nbinj_progcms))), df, on=id,all.y=TRUE)
  
  names(df) <- str_remove(names(df),'_num')
return(df)
}

excludeHosp_nbinj <- function(df,criterio){
  # we exclude hospitalizations due to birth, delivery or traumatologic injury
  filtros <- paste('nbinj_',criterio, sep='') 
  if (!(filtros %in% names(df))) stop(sprintf('Column %s is missing. Accepted values for argument "criterio" are: urg, prog, urgcms, progcms', filtros))

  anyfiltros=(df[filtros]>=1)
  crit=(df[criterio]>=1)
  
  df$remove=as.integer((anyfiltros & crit))
  
  df[criterio]=df[criterio]-df$remove
  return(df)
}

get_hospitalizations <- function(year_, X, column, exclude_nbinj=TRUE){
  # Función de nivel usuario. 
  # Devuelve los ingresos de tipo 'column' que tuvieron los pacientes contenidos X$PATIENT_ID en el año year_
  # Por defecto, excluye partos/nacimientos y traumatología
  # VALORES ACEPTABLES:
  #       year_: 2016, 2017, 2018
  #       X: El data.frame o data.table que contiene las variables clínicas (demográficas y/o CCS y/o fármacos)
  #       column: urg - Ingresos urgentes según criterios administrativos
  #               prog- Ingresos programados según criterios administrativos
  #               urgcms - Ingresos urgentes según criterio CMS
  #               progcms - Ingresos programados según criterio CMS
  # ATENCIÓN: La función devuelve el número de ingresos, no la presencia/ausencia. Para ello, hacer
  #           y <- get_hospitalizations(year_, X, column, exclude_nbinj) ;
  #           y_binaria <- as.integer(ing$urgcms>=1)) o la columna correspondiente
  ingT<-createYearlyDataFrames()
  print(names(ingT))
  print(year)
  ingT <- ingT[[year_]]
  ingT <- data.frame(ingT)
  ingT$PATIENT_ID <- ingT$id
  if (exclude_nbinj) ingT<-excludeHosp_nbinj(ingT,column)
  ingT <- ingT[, c('PATIENT_ID',column)]
  y <- merge(ingT,X[,'PATIENT_ID'],on='PATIENT_ID',all.y=TRUE)
  y[is.na(y)] <- 0
  return(y)
}