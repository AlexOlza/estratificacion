
##201612
i<-2 #Para 2017 poner 2 en lugar de 1
#indicar el camino a la carpeta que contiene los archivos de members y diagnósticos
datapath<-"/home/alex/Desktop/estratificacionDatos"
indispensabledatapath<-paste(datapath,'indispensable',sep='/')
#indicar dónde guardaremos los archivos inGMA
outpath <- '/home/alex/GMA_SNS_v30112022/dat'
#indicar cómo se llaman los archivos members
membersfile2016<-paste(indispensabledatapath,"members_201612.txt",sep='/')
membersfile2017<-paste(indispensabledatapath,"members_201712.txt",sep='/')
#indicar cómo se llaman los archivos de diagnósticos corregidos
dx2016 <- paste(indispensabledatapath,"ccs/gma_dx_in_2016.txt",sep='/')
dx2017 <- paste(indispensabledatapath,"ccs/gma_dx_in_2017.txt",sep='/')

#agrupamos los nombres de los archivos en vectores, usaremos unos u otros según el valor de i
members<-c(membersfile2016,membersfile2017)
dxyears <- c(dx2016, dx2017)
years<-c(2016,2017)

#Según el año i elegido, elegimos los archivos de entrada y creamos
#el nombre de los de salida
year<-years[i]
membersfile<- members[i]
dxyearfile <- dxyears[i]
outfile_h<-paste(outpath,sprintf("gma%s_h_in.txt", year),sep='/')
outfile_m<-paste(outpath,sprintf("gma%s_m_in.txt", year),sep='/')
#procesamos el archivo members
library(data.table)
membersyear<-fread(membersfile, header = FALSE, sep = ",")
colnames(membersyear)<-c("id","cupo","uap","osi","edad","sexo","coste","costefarmacia","urgencias","consultas","hosp","dial")
summary(membersyear)
library(dplyr)
#fechas de nacimiento doy ficticias para que den la edad
#
membersyear$nac<-year-membersyear$edad
membersyear$fecnac<-paste0(membersyear$nac,"-12-01")
membersyear$fecnac<-as.Date(membersyear$fecnac)
membersyear<-subset(membersyear,select=-c(nac))
#comprobacion
#members2016$edad2<-trunc(as.numeric(as.difftime(as.Date("2016-12-31")-members2016$fecnac),units="days")/365.25,0)
#members2016$dif<-members2016$edad2-members2016$edad
#sum(members2016$dif)


library(stringr)
#leemos los diagnósticos y nos quedamos con los id que están en members
gma2016<-fread(dxyearfile,sep = ",")
summary(gma2016)
names(gma2016)[1]<-"id"

gma2016<-gma2016 %>% filter(id %in% membersyear$id)

#preprocesamos para obtener el formato adecuado para GMA
gma2016<-merge(membersyear,gma2016,by="id", all=TRUE)
rm(membersyear)
summary(gma2016)
gma2016$CIE_VERSION<-as.factor(gma2016$CIE_VERSION)
levels(gma2016$CIE_VERSION)
gma2016$cie<-as.character(gma2016$CIE_VERSION)
gma2016$cie[gma2016$CIE_VERSION=="10"]<-2
gma2016$cie[gma2016$CIE_VERSION=="9"]<-1
gma2016$cie[gma2016$CIE_VERSION=="10CM"]<-5
gma2016$cie<-as.factor(gma2016$cie)
levels(gma2016$cie)

#FORMATO FECHAS (hago que todas las fechas de diagnostico sean de 20161231)
#gma2016$fecdx<-as.Date(gma2016$fecfin, "%Y-%m-%d")
#gma2016$fecdx<-format(gma2016$fecdx,"%Y%m%d")
gma2016$fecdx<-sprintf("%s1231",year)
gma2016$fecnac<-as.Date(gma2016$fecnac, "%Y-%m-%d")
gma2016$fecnac<-format(gma2016$fecnac,"%Y%m%d")
gma2016<-subset(gma2016,select=c(id,cie,CODE,fecdx,sexo,fecnac,uap))

##guardo la tabla
##parece que la manera de meterlo es trocearlo: hombres y mujeres
gma2016_h<-subset(gma2016,sexo==1)
gma2016_m<-subset(gma2016,sexo==2)
write.table(gma2016_h, outfile_h,na="", sep="|",row.names=FALSE,col.names=FALSE, quote = FALSE)
write.table(gma2016_m, outfile_m,na="", sep="|",row.names=FALSE,col.names=FALSE, quote = FALSE)

#Ahora los archivos de entrada están guardados donde deben
# CARGAMOS EL WORKSPACE DE GMA

load("/home/alex/GMA_SNS_v30112022/bin/GMA SNS v11.RData")

# Llamamos a la función AgrupaGMA

#Parámetros que no cambian para cada trozo de la población
path.in  = "/home/alex/GMA_SNS_v30112022/dat/"
path.out = "/home/alex/GMA_SNS_v30112022/dat/"
date.ini = "20000101"
date.end = sprintf("%s1231",year)
language = "ES"
#Parámetros que sí cambian (cambiar _h por _m para las mujeres)
file.in  = sprintf("gma%s_h_in.txt",year)
file.out = sprintf("outGMA_%s_h.txt",year)

# Llamada hombres
Agrupa.GMA(
  file.in,
  path.in,
  file.out,
  path.out,
  date.ini,
  date.end,
  language) 

# Llamada mujeres
file.in  = sprintf("gma%s_m_in.txt",year)
file.out = sprintf("outGMA_%s_m.txt",year)
Agrupa.GMA(
  file.in,
  path.in,
  file.out,
  path.out,
  date.ini,
  date.end,
  language) 


#ANÁLISIS DE LOS RESULTADOS PARA LOS HOMBRES
#Leemos el fichero de códigos no identificados 
no_ident<-fread(paste(path.out,sprintf('outGMA_%s_h_Miss.txt',year),sep='/'))
colnames(no_ident)<- c('CIE_VERSION','CODE','FREQUENCY')
summary(no_ident)
length(unique(no_ident$CODE))
#PARA 2016:
#Hay 718, de 6875378 (0.01%)
#Vemos que más del 75% de los códigos no identificados corresponden al CIE 10CM. 
#En general son códigos infrecuentes: el 3er cuantil de la frecuencia es 16.
#Hay alguno más frecuente, el máximo es 37167

#PARA 2017:
#Hay 559 de 7303118
#La mayoría del CIE10CM
#tercer cuantil de la frecuencia 11, máximo 19324


#Ahora analizamos las incongruencias
incongruencias<-fread(paste(path.out,sprintf('outGMA_%s_h_Inc.txt',year),sep='/'))
colnames(incongruencias)<-c('id','code','age','sex','inc_edad','inc_sexo')
summary(incongruencias)
length(unique(incongruencias$code))
#PARA 2016:
# Hay 15429 (603 códigos únicos)
# PARA 2017: 8954 (458)

#Ahora una pequeña exploración de la agrupación
agrupacion<-fread(paste(path.out,'outGMA_2016_h.txt',sep='/'))
colnames(agrupacion)<-c("id","zbs", "edad", "sexo","gma","num_patol","num_sist","peso-ip","riesgo-muerte","dm","ic","epoc","hta","depre","vih","c_isq","acv","irc","cirros","osteopor","artrosis",
                                  "artritis","demencia","dolor_cron","etiqueta","version")
summary(agrupacion)
cor(agrupacion$edad,agrupacion$`peso-ip`) # 0.3508647

library(corrplot)
corr<-round(cor(subset(agrupacion, select=c("edad","num_patol","num_sist","peso-ip","riesgo-muerte","dm","ic","epoc","hta","depre","vih","c_isq","acv","irc","cirros","osteopor","artrosis",
                                            "artritis","demencia","dolor_cron"))),2)
corrplot(corr, method="circle",mar=c(0,0,2,0),title=sprintf('Hombres %s',year))


#ANÁLISIS DE LOS RESULTADOS PARA LAS MUJERES
#Leemos el fichero de códigos no identificados 
no_ident<-fread(paste(path.out,sprintf('outGMA_%s_m_Miss.txt',year),sep='/'))
colnames(no_ident)<- c('CIE_VERSION','CODE','FREQUENCY')
summary(no_ident)
length(unique(no_ident$CODE))
#PARA 2016:
#Hay 590, de 8564133 (0.007%)
# PARA 2017: 449
#Vemos que más del 75% de los códigos no identificados corresponden al CIE 10CM. 
#En general son códigos infrecuentes: el 3er cuantil de la frecuencia es 12 (9 en 2017).
#Hay alguno más frecuente, el máximo es 31918 (16900 en 2017)

#Ahora analizamos las incongruencias
incongruencias<-fread(paste(path.out,sprintf('outGMA_%s_m_Inc.txt',year),sep='/'))
colnames(incongruencias)<-c('id','code','age','sex','inc_edad','inc_sexo')
summary(incongruencias)
length(unique(incongruencias$code))
#PARA 2016:
# Hay 17939 (526 códigos únicos)
# PARA 2017: 10243 (440)
#Ahora una pequeña exploración de la agrupación
agrupacion<-fread(paste(path.out,'outGMA_2016_m.txt',sep='/'))
colnames(agrupacion)<-c("id","zbs", "edad", "sexo","gma","num_patol","num_sist","peso-ip","riesgo-muerte","dm","ic","epoc","hta","depre","vih","c_isq","acv","irc","cirros","osteopor","artrosis",
                        "artritis","demencia","dolor_cron","etiqueta","version")
summary(agrupacion)
cor(agrupacion$edad,agrupacion$`peso-ip`) # 0.3721145

corr<-round(cor(subset(agrupacion, select=c("edad","num_patol","num_sist","peso-ip","riesgo-muerte","dm","ic","epoc","hta","depre","vih","c_isq","acv","irc","cirros","osteopor","artrosis",
                                            "artritis","demencia","dolor_cron"))),2)
corrplot(corr, method="circle",mar=c(0,0,2,0),title=sprintf('Mujeres %s',year))
