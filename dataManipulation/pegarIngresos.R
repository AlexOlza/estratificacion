library(data.table)
library(dplyr)
fname=c('estratificacionDatos/modelMarkersActivos/2016-2017Activos.csv',
        'estratificacionDatos/modelMarkersActivos/2017-2018Activos.csv',
        'estratificacionDatos/modelMarkersActivos/2018-2019Activos.csv')
iname=c('estratificacionDatos/ing1617.csv','estratificacionDatos/ing1718.csv','estratificacionDatos/ing1819.csv')
out=c("estratificacionDatos/ing2016-2017Activos.csv","estratificacionDatos/ing2017-2018Activos.csv",
      "estratificacionDatos/ing2018-2019Activos.csv")
for (i in 1:3){
  a=fread(fname[i])
  ing=fread(iname[i])
  full=left_join(a,ing,by='PATIENT_ID')
  # cols<-c('ingresoPrevioUrg','ingresoPrevioProg','ingresoUrg','ingresoProg')
  # full[cols][is.na(full[cols])] <- 0
  # x[c("a", "b")][is.na(x[c("a", "b")])] <- 0
  full[is.na(ingresoPrevioUrg), ingresoPrevioUrg:=0]
  full[is.na(ingresoPrevioProg), ingresoPrevioProg:=0]
  full[is.na(ingresoUrg), ingresoUrg:=0]
  full[is.na(ingresoProg), ingresoProg:=0]
  full=subset(full,select=-c(V1.x,V1.y))
  write.csv(full,out[i])
}
