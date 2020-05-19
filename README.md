# COVID-19


[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.es)

![2019_nCoV](https://github.com/Eclectikus/COVID-19/blob/master/img/2019_nCoV600x.png "2019_nCoV")

Datos públicos oficiales sobre la epidemia de Coronavirus COVID-19 recolectados para su proceso y visualización en **noSoloDatos**:

[**Seguimiento del COVID-19**](https://nosolodatos.netlify.com/es/covid19/coronavirus)

---

<<<<<<< HEAD
Para consultar las series temporales proporcionadas por el **Centro de Ciencia de Ingeniería y Sistemas de la Universidad Johns Hopkins** visita [este otro repositorio](https://github.com/Eclectikus/jhutimeseries).
=======
En este repositorio ([COVID-19/data/](https://github.com/Eclectikus/COVID-19/tree/master/data)) puedes descargar y/o consultar los siguientes archivos:

- **`PDFs`**: Todos los informes diarios originales publicados por el *`Ministerio de Sanidad`*:
  - *Actualizacion_xx_COVID-19.pdf* **>**  Desde la actualización **`37`** correspondiente al **`5`** de **`marzo`** a la **`99`** del **`8`** de **`mayo`**.
  - *Actualizacion_xxx_COVID-19.pdf* **>**  Desde la actualización **`100`** correspondiente al **`9`** de **`mayo`** a la última actualización disponible.

- **`CSVs`**: Los datos de incidencia por **`Comunidades Autónomas`** extraídos de los **`PDFs`** originales descritos en el punto anterior.
  - El nombre de los archivos sigue el siguiente formato: **`coviDDMM.csv`**

- **`Serie temporal`**: Este archivo se pude descargar o consultar en [COVID-19/data/st.csv](https://github.com/Eclectikus/COVID-19/blob/master/data/st.csv). Son los datos extraídos de las tablas anteriores, o más apropiadamente la compilación de todos ellos en un solo fichero (**`st.csv`**), que podría no estar en el formato ideal para su tratamiento dependiendo de la configuración de tu entorno de *software*. Por ejemplo para transformar el archivo a un formato amigable para procesar datos de *series de tiempo* en **`R`** se puede utilizar el siguiente código:

~~~~R
sturl <- "https://raw.githubusercontent.com/Eclectikus/COVID-19/master/data/st.csv"
stt <- read.csv(sturl, encoding = "UTF-8")

# Convierte en numéricos los valores del IA (importados originalmente como texto):
stt$IA <- as.numeric(as.character(stt$IA))

# Cambia la columna de fechas (originalmente texto) al formato de fecha utilizado por R:
stt$Fecha <- as.Date(as.character(stt$Fecha), "%d/%m/%y")

~~~~

---

<<<<<<< HEAD
Para consultar las series temporales proporcionadas por el **Centro de Ciencia de Ingeniería y Sistemas de la Universidad Johns Hopkins** visita [este otro repositorio](https://github.com/Eclectikus/jhutimeseries).
>>>>>>> e8dda7ff7ea4d941ecfb9a903c07d975a3c7265f
=======
El **`Mapa`** que se utiliza (en formato **`geojson`**) puede ser consultado o descargado en el directorio [COVID-19/mapa/](https://github.com/Eclectikus/COVID-19/tree/master/mapa). La codificación para las CCAA se puede ver en [COVID-19/mapa/codifCCAAgeojson.csv](https://github.com/Eclectikus/COVID-19/blob/master/mapa/codifCCAAgeojson.csv).

---

Para consultar las series temporales proporcionadas por el **Centro de Ciencia de Ingeniería y Sistemas de la Universidad Johns Hopkins** visita [este otro repositorio](https://github.com/Eclectikus/jhutimeseries).
>>>>>>> 7f1f91a0dc8cbc9d1c7fae593b682bfe7e310a2b
