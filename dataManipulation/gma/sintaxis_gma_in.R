
##201612
i<-1 #Para 2017 poner 2 en lugar de 1
datapath<-"/home/aolza/Desktop/estratificacionDatos"
indispensabledatapath<-paste(datapath,'indispensable',sep='/')
outpath <- '/home/aolza/GMA_SNS_v30112022/dat'
membersfile2016<-paste(indispensabledatapath,"members_201612.txt",sep='/')
membersfile2017<-paste(indispensabledatapath,"members_201712.txt",sep='/')

dx2016 <- paste(datapath,"gma_dx_in_2016.txt",sep='/')
dx2017 <- paste(datapath,"gma_dx_in_2017.txt",sep='/')

members<-c(membersfile2016,membersfile2017)
dxyears <- c(dx2016, dx2017)
years<-c(2016,2017)

year<-years[i]
membersfile<- members[i]
dxyearfile <- dxyears[i]
outfile_h<-paste(outpath,sprintf("gma%s_h_in.txt", year),sep='/')
outfile_m<-paste(outpath,sprintf("gma%s_m_in.txt", year),sep='/')

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

gma2016<-fread(dxyearfile,sep = ",")
summary(gma2016)
names(gma2016)[1]<-"id"

gma2016<-gma2016 %>% filter(id %in% membersyear$id)


gma2016<-merge(membersyear,gma2016,by="id") #all=TRUE esto es outer join! aparecen NAs
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
gma2016$fecdx<-"20161231"
gma2016$fecnac<-as.Date(gma2016$fecnac, "%Y-%m-%d")
gma2016$fecnac<-format(gma2016$fecnac,"%Y%m%d")
gma2016<-subset(gma2016,select=c(id,cie,CODE,fecdx,sexo,fecnac,uap))

##guardo la tabla
##parece que la manera de meterlo es trocearlo: hombres y mujeres
gma2016_h<-subset(gma2016,sexo==1)
gma2016_m<-subset(gma2016,sexo==2)
write.table(gma2016_h, outfile_h,na="", sep="|",row.names=FALSE,col.names=FALSE, quote = FALSE)
write.table(gma2016_m, outfile_m,na="", sep="|",row.names=FALSE,col.names=FALSE, quote = FALSE)


# CARGAMOS EL WORKSPACE DE GMA

load("/home/aolza/GMA_SNS_v30112022/bin/GMA SNS v11.RData")

# Llamamos a la función AgrupaGMA

#Parámetros que no cambian para cada trozo de la población
path.in  = "/home/aolza/GMA_SNS_v30112022/dat/"
path.out = "/home/aolza/GMA_SNS_v30112022/dat/"
date.ini = "20000101"
date.end = sprintf("%s1231",year)
language = "ES"
#Parámetros que sí cambian (cambiar _h por _m para las mujeres)
file.in  = "gma2016_h_in.txt"
file.out = sprintf("outGMA_%s_h.txt",year)

# Llamada
Agrupa.GMA(
  file.in,
  path.in,
  file.out,
  path.out,
  date.ini,
  date.end,
  language) 




















##