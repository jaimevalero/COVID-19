#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load data
get_ipython().system(' ls -latr ../data/*.csv')
# Añadimos un segundo remoto al original, para trabajar con los datos actualizados
get_ipython().system(' cd ..')
get_ipython().system(' git add . ')
get_ipython().system(' git commit -m "Refactor"')
get_ipython().system(' git remote add original git@github.com:Eclectikus/COVID-19.git')
get_ipython().system(' git pull -f original master')

get_ipython().system(" jupyter nbconvert --to script 'Loading data.ipynb'")


# In[6]:


# Load Data
import glob 
import pandas as pd

def Carga_All_Files( ):
    regexp='../data/covi*'
    df = pd.DataFrame()
    # Iterate trough LIST DIR and 
    for my_file in glob.glob(regexp):
        this_df = pd.read_csv(my_file)
        this_df['Fecha'] = my_file
        df = pd.concat([df,this_df])
    return df  

df = Carga_All_Files( )

df.tail()


# In[9]:


def Preprocesado():
    df = Carga_All_Files( )
    # Formateamos la fecha
    df['Fecha'].replace({
        '../data/covi': '2020-',
        '.csv'        : ''}, inplace=True, regex=True)
    df['Fecha'] =  pd.to_datetime(df['Fecha'], format='%Y-%d%m')
    # 
    
    return df.sort_values(by='Fecha')

df = Preprocesado()


# In[10]:


import numpy as np

def Get_Comunidad(comunidad):

    # Trabajamos solo con una comunidad
    df = Preprocesado()

    comunidad = df[(df['CCAA'] == COMUNIDAD_A_CONSIDERAR)].sort_values(by='Fecha')
    del comunidad['ID']
    del comunidad['IA']
    del comunidad['Nuevos']

    comunidad.set_index('Fecha', inplace=True)

    # Datos de fallecimientos diarios, en totales y tanto por uno.
    comunidad['Fallecidos hoy absoluto'] = comunidad['Fallecidos'] - comunidad['Fallecidos'].shift(1)
    comunidad['Fallecidos hoy porcentaje'] = comunidad['Fallecidos hoy absoluto']  / comunidad['Fallecidos'] 
    comunidad['Fallecidos hoy variacion diferencia'] = comunidad['Fallecidos hoy absoluto'] - comunidad['Fallecidos hoy absoluto'].shift(1)

    # Datos de Casos diarios,  en totales y tanto por uno.
    comunidad['Casos hoy absoluto'] = comunidad['Casos'] - comunidad['Casos'].shift(1)
    comunidad['Casos hoy porcentaje'] = comunidad['Casos hoy absoluto']  / comunidad['Casos'] 
    comunidad['Casos hoy variacion diferencia'] = comunidad['Casos hoy absoluto'] - comunidad['Casos hoy absoluto'].shift(1)
    # Convertimos a entero, para quitar decimales
    CONVERT_INT_COLUMNS = ['Fallecidos hoy absoluto', 
                           'Fallecidos hoy variacion diferencia',
                           'Casos hoy variacion diferencia',
                           'Casos hoy absoluto',
                           'Hospitalizados',
                           'Curados']
    for column in CONVERT_INT_COLUMNS :
        comunidad[column] = comunidad[column].fillna(0)
        comunidad[column] = comunidad[column].astype(np.int64)
    # ordenamos las filas y columnas
    columnsTitles = ['CCAA', 
                     'Casos'     , 'Casos hoy absoluto'     , 'Casos hoy variacion diferencia', 'Casos hoy porcentaje'      ,
                     'Fallecidos', 'Fallecidos hoy absoluto', 'Fallecidos hoy variacion diferencia', 'Fallecidos hoy porcentaje' ,
                     'Curados',
                     'UCI',  
                     'Hospitalizados']
    comunidad = comunidad.reindex(columns=columnsTitles)
    comunidad = comunidad.sort_values(by=['Fecha'], ascending=False)

    return comunidad


# In[12]:


COMUNIDAD_A_CONSIDERAR = 'Madrid'

comunidad = Get_Comunidad(COMUNIDAD_A_CONSIDERAR)
comunidad


# In[6]:


# Grafica

from matplotlib import pyplot as plt    

fig = plt.figure(figsize=(8, 6), dpi=80)
plt.plot( comunidad['Fallecidos hoy porcentaje'])
fig.suptitle('Crecimiento mortalidad '+COMUNIDAD_A_CONSIDERAR+' diario, en porcentaje', fontsize=20)
plt.ylabel('Fallecidos hoy, respecto al total', fontsize=16)

comunidad['Fallecidos hoy porcentaje']


# In[7]:


fig = plt.figure(figsize=(8, 6), dpi=80)
plt.plot(comunidad['Fallecidos hoy absoluto']) 
fig.suptitle('Mortalidad '+COMUNIDAD_A_CONSIDERAR+' diaria', fontsize=20)

comunidad['Fallecidos hoy absoluto']

