###############################################################################################
# Este fichero contiene las direcciones de los archivos de datos que se van a leer            #
# En el resto del código, llamaré a este fichero cuando necesite leer alguno de los archivos  #
# De esta forma, si alguno cambia de nombre o carpeta sólo hay que modificarlo aquí.          #
# Estructura recomendada de la carpeta de datos:                                              #
# CARPETA PRINCIPAL DATOS |                                                                   #
#                        .|Datos transformados (ej. tablas de los CCS)                        #
#                        .|__ CARPETA DATOS INDISPENSABLES |
#.                                                         |DATOS INDISPENSABLES RELACIONADOS CON LOS CCS
#.                                                         |          Al menos, los diagnosticos y los diccionarios CIE/CCS
###############################################################################################

########################################################################################
# Si otra persona está usando esto, que cambie el valor por defecto
# de carpeta_datos o use config <- configuration(carpeta_datos='direccion_de_carpeta')
#########################################################################################
library("stringr")
# función constructora de la clase "configuration"
configuration <- function(carpeta_datos ='/home/aolza/Desktop/estratificacionDatos',
                          # Ficheros de datos indispensables:
                          ficheros_dx =  c("2016"="dx_in_2016.txt","2017"="dx_in_2017.txt"),#esta variable será un diccionario con los años como claves 
                                                                                            #y los nombres de fichero como valores
                          ficheros_rx =   c("2016"="rx_in_2016.txt","2017"="rx_in_2017.txt"), #esta variable también
                          diccionario_cie9_ccs = 'translate_icd9_ccs_2015.csv',
                          diccionario_cie10cm_ccs = 'translate_icd10cm_ccs_2018.csv',
                          # Ficheros derivados de los datos indispensables:
                          ficheros_ccs = c("2016"="CCS2016.csv","2017"="CCS2017.csv")
                          ) {
  
  carpeta_datos_indispensable <- file.path(carpeta_datos,'indispensable')
  carpeta_ccs                 <- file.path(carpeta_datos_indispensable,'ccs')
  diccionario_cie9_ccs        <- file.path(carpeta_ccs, diccionario_cie9_ccs)
  diccionario_cie10cm_ccs        <- file.path(carpeta_ccs, diccionario_cie10cm_ccs)
  
  for (name in names(ficheros_dx)){
    # test de calidad
    if ( is.na(as.integer(name))) stop('ficheros_dx debe ser un diccionario con los años como claves. Ejemplo: \n
                                    ficheros_dx= c("2016"="dx_in_2016.txt","2017"="dx_in_2017.txt")')
    ficheros_dx[name]         <- file.path(carpeta_ccs,ficheros_dx[name])
  }
  
  for (name in names(ficheros_rx)){
    # test de calidad
    if ( is.na(as.integer(name))) stop('ficheros_rx debe ser un diccionario con los años como claves. Ejemplo: \n
                                    ficheros_rx= c("2016"="dx_in_2016.txt","2017"="dx_in_2017.txt")')
    ficheros_rx[name]         <- file.path(carpeta_ccs,ficheros_rx[name])
  }
  
  for (name in names(ficheros_ccs)){
    # test de calidad
    if ( is.na(as.integer(name))) stop('ficheros_ccs debe ser un diccionario con los años como claves. Ejemplo: \n
                                    ficheros_ccs= c("2016"="ccs_2016.txt","2017"="ccs_2017.txt")')
    ficheros_ccs[name]         <- file.path(carpeta_datos,ficheros_ccs[name])
  }
  
  value <- list(carpeta_datos=carpeta_datos,
                carpeta_datos_indispensable=carpeta_datos_indispensable,
                carpeta_ccs=carpeta_ccs,
                ficheros_dx=ficheros_dx, 
                ficheros_rx=ficheros_rx,
                diccionario_cie9_ccs=diccionario_cie9_ccs,
                diccionario_cie10cm_ccs=diccionario_cie10cm_ccs,
                ficheros_ccs = ficheros_ccs)
  attr(value, 'class') <- 'configuration'
  value
}

config <- configuration()