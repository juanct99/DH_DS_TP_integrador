import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
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

p, select = bokehLinePlot()

container = st.container()
container.bokeh_chart(column(p, select, sizing_mode = 'scale_width'))


def heatmap(df):
   df_heatmap = df.pivot_table(index="tipo_dia", columns="hora", values="pax_total", aggfunc=np.mean)
   fig, ax = plt.subplots(figsize=(12,6))
   fig.patch.set_facecolor(color)
   sns.heatmap(df_heatmap, cmap="YlOrRd", ax=ax)
   ax.yaxis.set_tick_params(rotation=0)
   ax.set_ylabel(None)
   ax.set_xlabel("Hora")
   plt.title("Pasajeros totales por tipo de día y hora",loc='left', fontsize=11)
   
   return fig

st.pyplot(heatmap(df_filtered))

def countplot(df,x,hue):
   fig, ax = plt.subplots(figsize=(12,6))
   sns.countplot(x=x, hue=hue, data=df, palette="YlOrRd", ax=ax)
   leg = ax.legend()
   for text in leg.get_texts():
    plt.setp(text, fontsize='14')
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
               palette = ['blue','red','green','#069AF3', 'yellow', 'purple'])

   plt.title('Media de pasajeros por linea', size=10)
   plt.xlabel("Pasajeros promedio", size=10)
   
   return fig

st.pyplot(media_pasajeros_linea(df_filtered))