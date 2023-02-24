import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, VArea
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import os


#------------------------ Data Analisys ------------------------#

st.set_page_config(page_title="Data analysis", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

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

#inicio escritura de la pagina

st.header("🔎Analisis exploratorio")

def metrics():
   cantidad_dfs = 29
   registros_iniciales = '~62M'
   cantidad_lineas_subte = 6
   cantidad_estaciones_subte = len(df.estacion.unique())

   st.write("")
   m1,m2,m3,m4 = st.columns(4, gap = "large")
   m1.metric("DataFrames", cantidad_dfs)
   m2.metric("Registros", registros_iniciales)
   m3.metric("Lineas", cantidad_lineas_subte)
   m4.metric("Estaciones", cantidad_estaciones_subte)
   st.write("")
   
metrics()

st.write("Luego del procesamiento y agrupamiento de los dfs, la estructura final resulta:")
st.dataframe(df.sample(5), use_container_width=True)



color = "#FDFFCD"
def bokehLinePlot():
   data_to_plot = df_filtered[['fecha','linea','pax_total']]
   data_to_plot = data_to_plot.set_index('fecha')
   data_to_plot.sort_values(by='fecha',ascending=True)
   y = data_to_plot['pax_total'].resample(grouped_by[group]).mean()

   test = pd.DataFrame({'total': y}).reset_index()
   dates = np.array(test['fecha'], dtype=np.datetime64)
   source = ColumnDataSource(data=dict(date=dates, close=test['total']))

   p = figure(height=250, width=800, tools="xpan", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above", x_range=(dates[5], dates[15]),
            background_fill_color=color, outline_line_color="black",
            border_fill_color = color)

   p.line('date', 'close', source=source, line_color="red", line_width=1.5)
   p.title.text = 'Pasajeros totales por ' + group.lower()
   p.title.text_font_size = '18px'
   p.ygrid.grid_line_color = 'grey'
   p.xgrid.grid_line_color = 'grey'
   
   
   start_span = np.datetime64('2020-03-20') # Inicio de la cuarentena
   end_span = np.datetime64('2021-09-22')
   source_span = ColumnDataSource({'x': [start_span, end_span],'y1': [0, 0],'y2': [420, 420]})
   p.add_glyph(source_span, VArea(x='x', y1='y1', y2='y2', fill_alpha=0.10, fill_color='red'))

   select = figure(height=80, width=800, y_range=p.y_range,
                  x_axis_type="datetime", y_axis_type=None,
                  tools="", toolbar_location=None, background_fill_color=color,
                  outline_line_color="black",
                  border_fill_color=color)

   range_tool = RangeTool(x_range=p.x_range)
   range_tool.overlay.fill_color = "navy"
   range_tool.overlay.fill_alpha = 0.2

   select.line('date', 'close', source=source, line_color="red", line_width=1.5)
   select.xgrid.grid_line_color = 'pink'

   select.add_glyph(source_span, VArea(x='x', y1='y1', y2='y2', fill_alpha=0.15, fill_color='red'))
   select.add_tools(range_tool)
   select.toolbar.active_multi = range_tool
   
   return p, select

p, select = bokehLinePlot()
container = st.container()
container.bokeh_chart(column(p, select, sizing_mode = 'scale_width'), use_container_width=True)

def heatmap(df):
   fig, ax = plt.subplots(figsize=(7,5))
   df_heatmap = df.pivot_table(index="tipo_dia", columns="hora", values="pax_total", aggfunc=np.mean)
   fig.patch.set_facecolor(color)
   sns.heatmap(df_heatmap, cmap="YlOrRd", ax=ax)
   ax.yaxis.set_tick_params(rotation=0)
   ax.xaxis.set_tick_params(rotation=0)
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

sns.set_style("ticks")
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

sns.set_style("ticks")
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

sns.set_style("ticks")
tabs = st.tabs(linea)
for i, tab in zip(linea, tabs):
    with tab:
       st.pyplot(media_pasajeros_estacion(df_filtered[df_filtered['linea'] == i]))
