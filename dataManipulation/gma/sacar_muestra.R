##urgencias, outpatients, dialisis 
##201612
setwd("F:/DSANI/3S-Osabide/EDUARDO/20160001")
members2016<-read.table("members_201612_101018.txt", header = FALSE, sep = ",")
summary(members2016)
colnames(members2016)<-c("id","cupo","uap","osi","edad","sexo","coste","costefarmacia","urgencias","consultas","hosp","dial")

library(dplyr)
members2016<-members2016 %>% mutate(n = row_number())

##cojo 150

members2016<-subset(members2016,n<151)


library(stringr)
setwd("F:/DSANI/3S-Osabide/EDUARDO/20160001")
dx2016<-read.table("dx_in_201612_p_pc_ca_co_u_di_do_cm_cmc_hs__131218.txt", header = FALSE, sep = ",")
summary(dx2016)
names(dx2016)[1]<-"id"
dx2016<-dx2016 %>% filter(id %in% members2016$id)
dx2016<-dx2016 %>% select(id,V2,V3,V12,V13,V16,V17)
names(dx2016)<-c("id","cie","codcie","fecini","fecfin","espe","lugar")


setwd("F:/DSANI/3S-Osabide/EDUARDO/gma/pruebas")
save(dx2016,file="dx2106prueba.rda")
save(members2016,file="members2106prueba.rda")

library(readr)
fecnac <- read_csv("fechanacimiento prueba.csv")
names(fecnac)<-c("id","fecnac")
summary(fecnac)
fecnac$fecnac<-as.Date(fecnac$fecnac)

##junto
gma<-merge(members2016,fecnac,by="id",all=TRUE)
summary(gma)

gma<-merge(gma,dx2016,by="id",all=TRUE)
summary(gma)
gma$cie<-as.factor(gma$cie)
levels(gma$cie)
gma$cie<-as.character(gma$cie)
gma$cie[gma$cie=="10"]<-2
gma$cie[gma$cie=="9"]<-1
gma$cie[gma$cie=="10CM"]<-5
gma$cie<-as.factor(gma$cie)
levels(gma$cie)

#FORMATO FECHAS
gma$fecdx<-as.Date(gma$fecfin, "%Y-%m-%d")
gma$fecdx<-format(gma$fecdx,"%Y%m%d")
gma$fecnac<-as.Date(gma$fecnac, "%Y-%m-%d")
gma$fecnac<-format(gma$fecnac,"%Y%m%d")
gma<-subset(gma,select=c(id,cie,codcie,fecdx,sexo,fecnac,uap))

##guardo la tabla
write.table(gma, "prueba_capv.txt",na="", sep="|",row.names=FALSE,col.names=FALSE, quote = FALSE)
summary(gma)
