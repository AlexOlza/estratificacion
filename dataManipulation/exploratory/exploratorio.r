#ANÁLISIS EXPLORATORIO BÁSICO DE LAS NUEVAS BBDD
#8Nov2021
#Proceso de toma de decisiones sobre cómo unificar los 12
#archivos que me ha pasado Edu

# ANOMALÍAS
#######################################################################################
#######################################################################################
#############################
#    REGISTROS ELIMINADOS   #
#############################
#Por ser duplicados:
#1) "procedimiento_mayor grd 2016.rda" 7158 registros
#2) "procedimiento_mayor grd 2017.rda" 7437
#3) "residenciado.rda" 9 
#4) "ingresos.rda" sum(duplicated(ing)) 8731
#############################
#   INGRESOS MUY LARGOS     # más de 365días
#############################

# summary(as(ing$fecalt-ing$fecing,"numeric"))#duracion de los ingresos
# sum(ing$fecalt-ing$fecing>365,na.rm=TRUE) #numero de ingresos largos
# summary(ing$fecalt[ing$fecalt-ing$fecing<=365])
#vemos que todos los ingresos de menos de 1 año fueron dados de alta antes de "2019-11-28" 
#         Min.      1st Qu.       Median         Mean      3rd Qu.         Max.         NA's 
# "2016-01-01" "2016-10-10" "2017-06-29" "2017-07-05" "2018-04-03" "2019-11-28"         "42" 
#######################################################################################
#######################################################################################

#############################
#   PREDICTORES NUEVOS      #
#############################
ficheros<-c('intubacion_2016.rda','intubacion_2017.rda',
            'procedimiento_mayor grd 2016.rda','procedimiento_mayor grd 2017.rda',
            'cirugia mayor 2016.rda','cirugia mayor 2017.rda',
            'dial_urgencias_consultas2016.rda','dial_urgencias_consultas2017.rda',
            'tratamiento cancer activo 2016.rda','tratamiento cancer activo 2017.rda',
            'residenciado.rda')
explorar<-function(fichero){
  print(fichero)
  df<-get(load(fichero))
  print(summary(df))
  print(any(duplicated(df$id)))#True=>Variable entera (varios eventos por paciente)
  print(sum(duplicated(df)))#registros duplicados (todas las columnas)
  }

for (f in ficheros){
  explorar(f)
}
#1)INTUBACIÓN- Vemos que sólo hay un marcador por paciente y año ->
#el archivo recoge que hubo intubación, no cuántas-> Var Binaria intubacion

#2) PROCEDIMIENTO MAYOR- Recoge cuántos px mayores -> Var entera pxmayor
#Variables px_mayor=Q y year sobran // La consideraremos como binaria

#3) CIRUGIA MAYOR- Sin duplicados -> Variable Binaria cirmayor
#Sobran tipo y year

#4) DIAL - Binaria. 
#5) URGENCIAS- Entera
#6) CONSULTAS- Entera

#7) CANCER- Binaria

#8) RESIDENCIADO- Sobra columna 'si'. Hay duplicados!
# Variable candidata a explorar. Cuidado: Recoge la situación en 2021!!!!!

#############################
#   REDEFINICION OUTCOME    #
#############################
ing<-get(load('ingresos2016_2018.rda'))#warning: tildes no UTF8 (ignorar)
#En las siguientes columnas los NA son en realidad ceros. Corrijo:
ing[c("planned_cms", "newborn_injury")][is.na(ing[c("planned_cms", "newborn_injury")])] <- 0
summary(ing)
#ing "sobra" pero se mantiene a propósito pq desde python haré un join (pd.merge) 
#cuando quiera juntar los ingresos con los ACG. Algo así:
#read predictors; read ingresos; dato=pd.join(pred, ing); datos.y=datos.y.fillna(0) 
#centro sobra, porque tenemos cod_centro (no la elimino)
sum(duplicated(ing$id))#como esperaba, algunas personas ingresan varias veces
sum(duplicated(ing$epi))#INESPERADO, episodios duplicados.->son registros completos duplicados, lo he comprobado
ing<-ing[!duplicated(ing),]
write.csv(ing,file = 'ingresos2016_2018.csv',sep=',',quote=FALSE,row.names = FALSE)
######################################
#   CONCLUSIONES- PLANNING OUTCOME   #
######################################
#La forma más eficiente de guardar la info implica tener varias tablas
#(ver: formas normales)
#pero esto será incómodo para trabajar en los modelos, así que voy a crear
#varios dataframes adecuados para los modelos que preveo hacer

#digresión: en el futuro tal vez habría que empezar
#a usar SQL para estas cosas... 
#(velocidad, tamaño, respeta las formas normales) 
#pero pandas es cómodo, legible, bien integrado en python

#1) 3 TABLAS CON LAS VARIABLES RESPUESTA (2016,2017,2018), 51 COLUMNAS
#política: desde python leeremos sólo las necesarias para cada modelo

#id|cod_centro|sin_alta|

#las siguientes variables no discriminan h.dia ni newborn_injury

#dias_ing_urg|dias_ing_prog|dias_ing_urgcms|dias_ing_progcms|
#num_ing_urg|num_ing_prog|num_ing_urgcms|num_ing_progcms

#anoprevio_dias_ing_urg|anoprevio_dias_ing_prog|anoprevio_dias_ing_urgcms|anoprevio_dias_ing_progcms|
#anoprevio_num_ing_urg|anoprevio_num_ing_prog|anoprevio_num_ing_urgcms|anoprevio_num_ing_progcms

#Se añaden los mismos 2 bloques de variables pero contando sólo ingresos en h.dia
#Esto permitirá hacer la resta para excluir estos ingresos, tanto en número como en días.
#sintaxis: hdia_col donde col son todas las anteriores

#idem para newborn_injury. sintaxis: newborninj_col

#política: Cálculo de los días de ingreso
#si el alta es antes del 31Dic del mismo año: fecalt-fecing,sin_alta=0
#si no, 31Dic-fecing,sin_alta=1
#si el alta es NA, dos opciones:
#       A) Asumir que sigue ingresado: 31Dic-fecing,sin_alta=1
#       B) Imputar: la mediana en el centro y teniendo en cuenta el tipo de ingreso,sin_alta=?

