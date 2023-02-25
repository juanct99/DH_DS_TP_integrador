import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, VArea
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, show

#------------------------ The model ------------------------#

st.set_page_config(page_title="The model", page_icon='🕰️',
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

df = read_file(path)

st.dataframe(df.head())

Linea_MaskA = df['linea'] == 'LineaA'
Linea_MaskB = df['linea'] == 'LineaB'
Linea_MaskC = df['linea'] == 'LineaC'
Linea_MaskD = df['linea'] == 'LineaD'
Linea_MaskE = df['linea'] == 'LineaE'
Linea_MaskH = df['linea'] == 'LineaH'

Data_test_A = df.loc[Linea_MaskA]
Data_test_B = df.loc[Linea_MaskB]
Data_test_C = df.loc[Linea_MaskC]
Data_test_D = df.loc[Linea_MaskD]
Data_test_E = df.loc[Linea_MaskE]
Data_test_H = df.loc[Linea_MaskH]


def agrupacion(dfinput):

    data_new = dfinput
    
    Suma_mes_test = data_new.groupby(by=['fecha','linea','tipo_dia'])['pax_total'].sum().reset_index()

    Suma_mes_test['fecha'] = pd.to_datetime(Suma_mes_test['fecha'], dayfirst=True)   

    data_test1 = Suma_mes_test.set_index('fecha')

    data_test1.sort_values(by='fecha',ascending=False)

    y = data_test1['pax_total'].resample('M').sum()

    Practica = pd.DataFrame({'pax_total': y}).reset_index()

    Practica.index = pd.PeriodIndex(Practica['fecha'], freq='M')
        
    return Practica
  

Practica = agrupacion(df)
PracticaA = agrupacion(Data_test_A)
PracticaB = agrupacion(Data_test_B)
PracticaC = agrupacion(Data_test_C)
PracticaD = agrupacion(Data_test_D)
PracticaE = agrupacion(Data_test_E)
PracticaH = agrupacion(Data_test_H)

Listd = [Practica,PracticaA,PracticaB,PracticaC,PracticaD,PracticaE,PracticaH]

def bokehlineplot2(List):
  
  p = figure(width=800, height=250, x_axis_type="datetime")
  p.title.text = 'Click on legend entries to mute the corresponding lines'
  for data, name, color in zip(List, ["total","lineaA","lineaB","lineaC","lineaD","lineaE","lineaH" ], Spectral6):
    df = pd.DataFrame(data)
  df['fecha'] = pd.to_datetime(df['fecha'])
  p.line(df['fecha'], df['pax_total'], line_width=2, color=color, alpha=0.8,
                 muted_color=color, muted_alpha=0.2, legend_label=name)
  p.legend.location = "top_left"
  p.legend.click_policy="mute"
  show(p)
  return p

st.pyplot(bokehlineplot2(Listd))


# evaluate = st.sidebar.checkbox(
#   "Evaluate my model", value=True)


 
