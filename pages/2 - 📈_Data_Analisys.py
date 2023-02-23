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

#------------------------ Data Analisys ------------------------#

st.set_page_config(page_title="Data analisys", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

df = read_file("../data/dfs_day_grouped.csv")
df.fecha = pd.to_datetime(df.fecha, format='%Y-%m-%d')

grouped_by = {
   "Día": "D",
   "Semana": "W",
   "Mes": "MS"
}

c1,c2 = st.columns(2)
with c1:
   lineas = ["TODAS"] + df.linea.unique().tolist()
   linea = st.selectbox('Linea',lineas)
with c2:
   group = st.selectbox('Agrupar por',list(grouped_by.keys()), index=2)

data_to_plot = df[['fecha','linea','pax_total']] if linea == 'TODAS' else df[df.linea == linea][['fecha','linea','pax_total']]
data_to_plot = data_to_plot.set_index('fecha')
data_to_plot.sort_values(by='fecha',ascending=True)
y = data_to_plot['pax_total'].resample(grouped_by[group]).mean()

test = pd.DataFrame({'total': y}).reset_index()
dates = np.array(test['fecha'], dtype=np.datetime64)
source = ColumnDataSource(data=dict(date=dates, close=test['total']))

color = "#FDFFCD"
p = figure(height=300, width=800, tools="xpan", toolbar_location=None,
           x_axis_type="datetime", x_axis_location="above", x_range=(dates[50], dates[70]),
           background_fill_color=color, outline_line_color="black",
           border_fill_color = color)

p.line('date', 'close', source=source)
p.yaxis.axis_label = 'Total pasajeros'

select = figure(height=130, width=800, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color=color,
                outline_line_color="black",
                border_fill_color=color)

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
select.line('date', 'close', source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool

container = st.container()
container.bokeh_chart(column(p, select, sizing_mode = 'scale_width'))