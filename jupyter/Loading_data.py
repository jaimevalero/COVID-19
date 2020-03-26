#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load data
get_ipython().system(' ls -latr ../data/*.csv')
# Añadimos un segundo remoto al original, para trabajar con los datos actualizados
get_ipython().system(' cd ..')
get_ipython().system(' git add . ')
get_ipython().system(' git commit -m "Refactor"')
get_ipython().system(' git remote add original git@github.com:Eclectikus/COVID-19.git')
get_ipython().system(' git pull -f original master')

get_ipython().system(" jupyter nbconvert --to script 'Loading_data.ipynb'")


# In[2]:


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


# In[3]:


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


# In[4]:


import numpy as np

def Enrich_Columns(comunidad):
    del comunidad['ID']
    del comunidad['IA']
    del comunidad['Nuevos']

    if 'Fecha' in comunidad.columns :
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
    comunidad = comunidad.rename(columns = {'CCAA':'Lugar'})

    return comunidad

def Get_Comunidad(nombre_comunidad):
    # Trabajamos solo con una comunidad
    df = Preprocesado()
    df = df[(df['CCAA'] == nombre_comunidad)].sort_values(by='Fecha')
    df = Enrich_Columns(df)
    return df

def Get_Nacion():
    df = Preprocesado()
    df = df.sort_values(by='Fecha')
    df = df.groupby(['Fecha']).sum()
    df['CCAA'] = 'España'
    df = Enrich_Columns(df)
    return df


# In[5]:


# Just for debug purposes
def Debug_Get_Comunidad():
    COMUNIDAD_A_CONSIDERAR = 'Madrid'
    comunidad = Get_Comunidad('Madrid')
    return comunidad

Debug_Get_Comunidad()


# In[6]:


# Just for debug purposes
def Debug_Get_Nacion():
    return Get_Nacion()

Debug_Get_Nacion()


# In[ ]:





# In[7]:


# Grafica

from matplotlib import pyplot as plt    

fig = plt.figure(figsize=(8, 6), dpi=80)
plt.plot( comunidad['Fallecidos hoy porcentaje'])
fig.suptitle('Crecimiento mortalidad '+COMUNIDAD_A_CONSIDERAR+' diario, en porcentaje', fontsize=20)
plt.ylabel('Fallecidos hoy, respecto al total', fontsize=16)

comunidad['Fallecidos hoy porcentaje']


# In[ ]:


fig = plt.figure(figsize=(8, 6), dpi=80)
plt.plot(comunidad['Fallecidos hoy absoluto']) 
fig.suptitle('Mortalidad '+COMUNIDAD_A_CONSIDERAR+' diaria', fontsize=20)

comunidad['Fallecidos hoy absoluto']

