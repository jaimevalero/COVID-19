[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.es)

![2019_nCoV](https://github.com/Eclectikus/COVID-19/blob/master/img/2019_nCoV600x.png "2019_nCoV")

# COVID-19

Datos públicos oficiales sobre la epidemia de Coronavirus COVID-19 recolectados para su proceso y visualización en **`noSoloDatos`**:

- [**Seguimiento del COVID-19**](https://nosolodatos.netlify.com/es/covid19/coronavirus). Esta página recopila los datos oficiales principales (casos confirmados, fallecidos, hospitalizados, UCIs...) por CCAA entre el **`5 de marzo de 2020`** y el **`5 de junio de 2020`**.
- [**Series temporales COVID-19**](https://eclectikus.github.io/jhutimeseries/). Esta página muestra de una manera accesible los datos de **`casos confirmados`**, **`fallecidos`** y **`recuperados`** en el mundo, proporcionados por el *Centro de Ciencia de Ingeniería y Sistemas* de la *Universidad Johns Hopkins* ([repo](https://github.com/CSSEGISandData/COVID-19)).
- [**Estudio de Seroprevalencia `ENE-Covid19`**](https://nosolodatos.netlify.app/es/covid19/seroprev). **`Ronda 1`** - **`Ronda 2`**. Esta página recopila los resultados del "*Estudio Nacional de sero-Epidemiología de la infección por SARS-CoV-2 en España (ENE-Covid19)*" del **Instituto de Salud Carlos III** ([página oficial del estudio](https://portalcne.isciii.es/enecovid19/)). [**`Trabajo en proceso`**]
- [**Cronología del `SARS-CoV-2`**](https://nosolodatos.netlify.app/es/covid19/crono). Cronología de los principales hitos de la pandemia. [**`Trabajo en proceso`**]

---

## Datos

### Incidencia por Comunidades Autónomas

En este repositorio ([COVID-19/data/](https://github.com/Eclectikus/COVID-19/tree/master/data)) se pueden descargar y/o consultar los siguientes archivos:

- **`PDFs`**: Todos los informes diarios originales publicados por el *`Ministerio de Sanidad`*:
  - *Actualizacion_xx_COVID-19.pdf* **>**  Desde la actualización **`37`** correspondiente al **`5`** de **`marzo`** a la **`99`** del **`8`** de **`mayo`**.
  - *Actualizacion_xxx_COVID-19.pdf* **>**  Desde la actualización **`100`** correspondiente al **`9`** de **`mayo`** a la última actualización disponible.

- **`CSVs`**: Los datos de incidencia por **`Comunidades Autónomas`** extraídos de los **`PDFs`** originales descritos en el punto anterior.
  - El nombre de los archivos sigue el siguiente formato: **`coviDDMM.csv`**

- **`Serie temporal`**: Este archivo se pude descargar o consultar en [COVID-19/data/st.csv](https://github.com/Eclectikus/COVID-19/blob/master/data/st.csv). Son los datos extraídos de las tablas anteriores, o más apropiadamente la compilación de todos ellos en un solo fichero (**`st.csv`**), que podría no estar en el formato ideal para su tratamiento dependiendo de la configuración de tu entorno de *software*. Por ejemplo para transformar el archivo a un formato amigable para procesar datos de *series de tiempo* en **`R`** se puede utilizar el siguiente código:

~~~R
sturl <- "https://raw.githubusercontent.com/Eclectikus/COVID-19/master/data/st.csv"
stt <- read.csv(sturl, encoding = "UTF-8")

# Convierte en numéricos los valores del IA (importados originalmente como texto):
stt$IA <- as.numeric(as.character(stt$IA))

# Cambia la columna de fechas (originalmente texto) al formato de fecha utilizado por R:
stt$Fecha <- as.Date(as.character(stt$Fecha), "%d/%m/%y")

~~~

### Seroprevalencia

**Estudio Nacional de sero-Epidemiología de la infección por SARS-CoV-2 en España (ENE-Covid19)**

- **`PDFs` originales**
  - **`Ronda 1` (13 de mayo)** **`>`** [*ene_covid19_inf_pre.pdf*](https://github.com/Eclectikus/COVID-19/blob/master/data/ene_covid19_inf_pre.pdf)
  - **`Ronda 2` (3 de junio)** **`>`** [*ene_covid19_inf_pre2.pdf*](https://github.com/Eclectikus/COVID-19/blob/master/data/ene_covid19_inf_pre2.pdf)

- **Estudio de seroprevalencia `ENE-Covid19`**: Datos obtenidos del repositorio de [**Datadista**](https://github.com/datadista/datasets/tree/master/COVID%2019):  
  - **`Ronda 1`** (13 de mayo de 2020) **`>`** [COVID-19/data/prevalencia1.csv](https://github.com/Eclectikus/COVID-19/blob/master/data/prevalencia1.csv)
  - **`Ronda 1`** (Act. 3 de junio de 2020) **`>`** [COVID-19/data/prevalencia1_2.csv](https://github.com/Eclectikus/COVID-19/blob/master/data/prevalencia1_2.csv)
  - **`Ronda 2`** (3 de junio de 2020) **`>`** [COVID-19/data/prevalencia2_1.csv](https://github.com/Eclectikus/COVID-19/blob/master/data/prevalencia2_1.csv)

---

## Mapas

Los **`Mapas`** utilizados (en formato **`geojson`**) pueden ser consultados o descargados en el directorio [COVID-19/map/](https://github.com/Eclectikus/COVID-19/tree/master/map).

- **`Comunidades Autónomas`**: [COVID-19/map/ccaa.geojson](https://github.com/Eclectikus/COVID-19/blob/master/map/ccaa.geojson).
  - La codificación de la toponimia para las *CCAA* se puede consultar en [COVID-19/map/codifCCAAgeojson.csv](https://github.com/Eclectikus/COVID-19/blob/master/map/codifCCAAgeojson.csv).
- **`Provincias`**: [COVID-19/map/prov.geojson](https://github.com/Eclectikus/COVID-19/blob/master/map/prov.geojson).
  - La codificación de la toponimia para las *provincias* se puede consultar en [COVID-19/map/ 	codifPROVgeojson.csv](https://github.com/Eclectikus/COVID-19/blob/master/map/codifPROVgeojson.csv). Aunque se utilizan los topónimos en castellano, la nomenclatura en las lenguas cooficiales se incluye también en el fichero **`geojson`**.

---

## Datos globales

Para consultar las series temporales proporcionadas por el **Centro de Ciencia de Ingeniería y Sistemas de la Universidad Johns Hopkins** visita [este otro repositorio](https://github.com/Eclectikus/jhutimeseries) y/o [esta página](https://eclectikus.github.io/jhutimeseries/).
