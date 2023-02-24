import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.wkt
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
import os

data_location = 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.csv'
barrios_data_raw = pd.read_csv(data_location, sep=';')
barrios_data_clean = barrios_data_raw.loc[:,('WKT','BARRIO')]
barrios_data_clean["WKT"] = barrios_data_clean["WKT"].apply(shapely.wkt.loads) 
geo_barrios = gpd.GeoDataFrame(barrios_data_clean, geometry='WKT',crs=3857)

estaciones_subte = gpd.read_file("data/estacionesdesubte.geojson")
geo_subte_new = gpd.GeoDataFrame(estaciones_subte, geometry = 'geometry' ,crs=3857)

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
lineas_subte = gpd.read_file("data/reddesubterraneo.kml", driver='KML')

#---Graf Subtes2--#

def GrafSubtes2():
  fig, ax = plt.subplots(figsize=(15, 10))
  plt.grid()

  geo_barrios.plot(ax=ax,color='grey')
   
  geo_subte_h = geo_subte_new.loc[geo_subte_new['LINEA'] =='H', :]
  geo_subte_h.geometry.plot(ax=ax, color ='yellow', label = 'Linea H')
  annotations = ['CASEROS',
   'INCLAN',
   'HUMBERTO 1°',
   'VENEZUELA',
   'ONCE - 30 DE DICIEMBRE',
   'CORRIENTES',
   'PARQUE PATRICIOS',
   'HOSPITALES',
   'CÓRDOBA',
   'LAS HERAS',
   'SANTA FE']
  x = [-58.39893,-58.40097,-58.40232,-58.40473, -58.40604, -58.40545, -58.40579, -58.41239, -58.40372, -58.39722,-58.40233]
  y = [-34.63575, -34.62938, -34.62309, -34.61524, -34.60894, -34.60449, -34.63841, -34.64127, -34.59846, -34.58746, -34.59440]
  for xi, yi, text in zip(x,y,annotations):
      ax.annotate(text, xy=(xi,yi), size = 5)
  #ax.annotate('CASEROS', xy=(-58.39893, -34.63575))

  geo_subte_a = geo_subte_new.loc[geo_subte_new['LINEA'] =='A', :]
  geo_subte_a.geometry.plot(ax=ax, color ='lightskyblue', label = 'Linea A')
  annotations1 = ['PERU',
   'PIEDRAS',
   'LIMA',
   'SAENZ PEÑA',
   'CONGRESO',
   'PASCO',
   'ALBERTI',
   'PLAZA DE MISERERE',
   'LORIA',
   'CASTRO BARROS',
   'RIO DE JANEIRO',
   'ACOYTE',
   'PRIMERA JUNTA',
   'PLAZA DE MAYO',
   'CARABOBO',
   'PUAN',
   'SAN PEDRITO',
   'SAN JOSÉ DE FLORES']
  x2 = [-58.37427,-58.37909,-58.38223,-58.38678,-58.39267,-58.39843, -58.40121,-58.40671,-58.41519, -58.42182, -58.42950, -58.43643, -58.44118,
  -58.37097,-58.45671,-58.44865,-58.46964,-58.46354]
  y2 = [-34.60856,-34.60888, -34.60910,-34.60941, -34.60923, -34.60965, -34.60983, -34.60982, -34.61078, -34.61177, -34.61521, -34.61828, -34.62041,-34.60881,
  -34.62667,-34.62353,-34.63071,-34.62909]
  for xi2, yi2, text2 in zip(x2,y2,annotations1):
      ax.annotate(text2, xy=(xi2,yi2), size = 5, rotation = 'vertical' )

  x3 = [-58.37816, -58.37953, -58.38061, -58.38044, -58.38017, -58.38143, -58.37782, -58.37992]
  y3 = [-34.60177, -34.60484,-34.60898,-34.61262,-34.61813,-34.62762,
  -34.59506, -34.62192]
  annotations2 = ['RETIRO',
   'LAVALLE',
   'DIAGONAL NORTE',
   'AV. DE MAYO',
   'MORENO',
   'INDEPENDENCIA',
   'CONSTITUCION',
   'SAN MARTIN',
   'SAN JUAN']
  geo_subte_c = geo_subte_new.loc[geo_subte_new['LINEA'] =='C', :]
  geo_subte_c.geometry.plot(ax=ax, color ='blue', label = 'Linea C')
  for xi3, yi3, text3 in zip(x3,y3,annotations2):
      ax.annotate(text3, xy=(xi3,yi3), size = 5 )
  x4 = [-58.48639]
  y4 = [-34.57432]
  annotations3 = ['JUAN MANUEL DE ROSAS']
  geo_subte_b = geo_subte_new.loc[geo_subte_new['LINEA'] =='B', :]
  geo_subte_b.geometry.plot(ax=ax, color ='red', label = 'Linea B')
  for xi4, yi4, text4 in zip(x4,y4,annotations3):
      ax.annotate(text4, xy=(xi4,yi4), size = 5, rotation = 'vertical' )

  x5 = [-58.46165]
  y5 = [-34.64331]

  annotations4 = [
   'PLAZA DE LOS VIRREYES']
  geo_subte_e = geo_subte_new.loc[geo_subte_new['LINEA'] =='E', :]
  geo_subte_e.geometry.plot(ax=ax, color ='purple', label = 'Linea B')
  for xi5, yi5, text5 in zip(x5,y5,annotations4):
      ax.annotate(text5, xy=(xi5,yi5), size = 5 )

  x6 = [-58.46238]
  y6 = [-34.55564]
  annotations5 = [
   'CONGRESO DE TUCUMAN']
  geo_subte_d = geo_subte_new.loc[geo_subte_new['LINEA'] =='D', :]
  geo_subte_d.geometry.plot(ax=ax, color ='green', label = 'Linea D')
  for xi6, yi6, text6 in zip(x6,y6,annotations5):
      ax.annotate(text6, xy=(xi6,yi6), size = 5, rotation = 'vertical' )

  lineas_subte.plot(ax=ax, color = 'black')
  ax.legend()

  return fig
#---Graf Subtes2--#
st.pyplot(GrafSubtes2())