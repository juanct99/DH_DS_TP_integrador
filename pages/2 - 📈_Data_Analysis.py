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


#------------------------ Data Analisys ------------------------#

st.set_page_config(page_title="Data analysis", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")
#---No Esta tomando GeoPandas---#
data_location = 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.csv'
barrios_data_raw = pd.read_csv(data_location, sep=';')
barrios_data_clean = barrios_data_raw.loc[:,('WKT','BARRIO')]
barrios_data_clean["WKT"] = barrios_data_clean["WKT"].apply(shapely.wkt.loads) 
geo_barrios = gpd.GeoDataFrame(barrios_data_clean, geometry='WKT',crs=3857)

estaciones_subte = gpd.read_file("data/estacionesdesubte.geojson")
geo_subte_new = gpd.GeoDataFrame(estaciones_subte, geometry = 'geometry' ,crs=3857)

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
lineas_subte = gpd.read_file("data/reddesubterraneo.kml", driver='KML')

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

df = read_file(path)
df.fecha = pd.to_datetime(df.fecha, format='%Y-%m-%d')
df.hora = df.hora.astype(int)

grouped_by = {
   "Día": "D",
   "Semana": "W",
   "Mes": "MS"
}
tipo_dia_dict = {"H": "Habiles",
                 "F": "Feriados",
                 "S": "Sabados",
                 "D": "Domingos"}

st.sidebar.subheader("Filtros")
with st.sidebar.expander("Años", expanded=False):
   year_options = [2017,2018,2019,2020,2021,2022]
   years = st.multiselect("Años seleccionados",year_options,default=year_options)

with st.sidebar.expander("Tipo de día", expanded=False):
   tipo_dia = df.tipo_dia.unique().tolist()
   tipo_dia = st.multiselect("Tipos de día seleccionados",
                             tipo_dia,
                              default=tipo_dia,
                              format_func=lambda x: tipo_dia_dict.get(x))
   
with st.sidebar.expander("Lineas", expanded=False):
   lineas = df.linea.unique().tolist()
   linea = st.multiselect("Lineas incluidas",lineas,
                           default=lineas,
                           format_func=lambda x: x.replace('Linea', ''))

with st.sidebar.expander("Rango horario", expanded=False):
   min_hour = 0
   max_hour = 23
   from_hour, to_hour = st.slider(
    "Selecciona un rango horario:",
    min_value=min_hour,
    max_value=max_hour,
    value=(min_hour, max_hour))
   
with st.sidebar.expander("Agrupamiento", expanded=False):
   group = st.radio('Seleccionar temporalidad:',list(grouped_by.keys()), index=2,
                     help = """
                     Agrupa los datos por día, semana o mes solo en los graficos de lineas temporales
                     """)



year_mask = df.fecha.dt.year.isin(years)
tipo_dia_mask = df.tipo_dia.isin(tipo_dia)
line_mask = df.linea.isin(linea)
horas_mask = (df.hora >= from_hour) & (df.hora <= to_hour)
df_filtered = df[line_mask & year_mask & tipo_dia_mask & horas_mask]


color = "#FDFFCD"
def bokehLinePlot():
   data_to_plot = df_filtered[['fecha','linea','pax_total']]
   data_to_plot = data_to_plot.set_index('fecha')
   data_to_plot.sort_values(by='fecha',ascending=True)
   y = data_to_plot['pax_total'].resample(grouped_by[group]).mean()

   test = pd.DataFrame({'total': y}).reset_index()
   dates = np.array(test['fecha'], dtype=np.datetime64)
   source = ColumnDataSource(data=dict(date=dates, close=test['total']))

   p = figure(height=300, width=800, tools="xpan", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above", x_range=(dates[5], dates[15]),
            background_fill_color=color, outline_line_color="black",
            border_fill_color = color)

   p.line('date', 'close', source=source, line_color="red", line_width=1.5)
   p.title.text = 'Pasajeros totales por ' + group.lower()
   p.title.text_font_size = '18px'
   p.ygrid.grid_line_color = 'grey'
   p.xgrid.grid_line_color = 'grey'

   select = figure(height=130, width=800, y_range=p.y_range,
                  x_axis_type="datetime", y_axis_type=None,
                  tools="", toolbar_location=None, background_fill_color=color,
                  outline_line_color="black",
                  border_fill_color=color)

   range_tool = RangeTool(x_range=p.x_range)
   range_tool.overlay.fill_color = "navy"
   range_tool.overlay.fill_alpha = 0.2

   select.line('date', 'close', source=source, line_color="red", line_width=1.5)
   select.xgrid.grid_line_color = 'pink'

   select.add_tools(range_tool)
   select.toolbar.active_multi = range_tool
   
   return p, select
  
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
  plt.show();
  return fig
#---Graf Subtes2--#

st.pyplot(GrafSubtes2)

p, select = bokehLinePlot()

container = st.container()
container.bokeh_chart(column(p, select, sizing_mode = 'scale_width'))

sns.set(font_scale=1)

def heatmap(df):
   fig, ax = plt.subplots(figsize=(7,5))
   df_heatmap = df.pivot_table(index="tipo_dia", columns="hora", values="pax_total", aggfunc=np.mean)
   fig.patch.set_facecolor(color)
   sns.heatmap(df_heatmap, cmap="YlOrRd", ax=ax)
   ax.yaxis.set_tick_params(rotation=0)
   ax.set_ylabel(None)
   ax.set_xlabel("Hora")
   plt.title("Pasajeros totales por tipo de día y hora",loc='left')
   sns.set(font_scale=1)
   return fig

st.pyplot(heatmap(df_filtered))

def countplot(df,x,hue):
   fig, ax = plt.subplots(figsize=(7,5))
   sns.countplot(x=x, hue=hue, data=df, palette="YlOrRd", ax=ax)
   leg = ax.legend()
   for text in leg.get_texts():
     plt.setp(text)
   ax.set_ylabel(None)
   ax.set_xlabel(None)
   ax.tick_params(axis='both', which='major', labelsize=16)
   fig.patch.set_facecolor(color)
    
   return fig


c3,c4 = st.columns(2)
with c3:
   st.pyplot(countplot(df_filtered,'linea', 'sentido'))
with c4:
   st.pyplot(countplot(df_filtered,'linea', 'tipo_dia'))
      
    
def media_pasajeros_linea(df):

   totalxlinea = df.groupby(by=['linea','hora'])['pax_total']
   totalxlinea = pd.DataFrame(totalxlinea.aggregate([np.min, np.median, np.mean, np.max]).round(2))
   totalxlinea.sort_values(by='mean', ascending=False, inplace=True)
   totalxlinea.reset_index(inplace=True)

   fig, ax = plt.subplots(figsize=(7,5))
   fig.patch.set_facecolor(color)
   sns.barplot(x = totalxlinea['mean'], 
               y=totalxlinea['linea'], 
               palette = "YlOrRd")

   ax.set_ylabel(None)
   sns.set(font_scale=1)
   plt.title('Media de pasajeros por linea', size=10)
   plt.xlabel("Pasajeros promedio", size=7)
   
   return fig

sns.set(font_scale=1)  
st.pyplot(media_pasajeros_linea(df_filtered))


def media_pasajeros_estacion(df):
   df = df.groupby(by=['linea','estacion','sentido','hora'])['pax_total']
   df = pd.DataFrame(df.aggregate([np.min, np.median, np.mean, np.max]).round(2))
   df = df.sort_values(by='mean', ascending=False)
   df.reset_index(inplace=True)
   
   fig, ax = plt.subplots(figsize=(7,5))
   fig.patch.set_facecolor(color)
   
   sns.barplot(x = df['amax'], 
               y=df['estacion'], palette = 'YlOrRd')

   plt.title('Media de pasajeros por estación', size=10)
   plt.ylabel(None)
   plt.yticks(size=5)
   plt.xticks(size=5)
   plt.xlabel("Pasajeros promedio", size=5)
   
   return fig

tabs = st.tabs(linea)
for i, tab in zip(linea, tabs):
    with tab:
        st.pyplot(media_pasajeros_estacion(df_filtered[df_filtered['linea'] == i]))
