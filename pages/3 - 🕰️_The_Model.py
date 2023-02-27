import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy.typing
import geopandas as gpd
import shapely.wkt

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, VArea
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral7
from bokeh.plotting import figure, show

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot

#------------------------ The model Intro ------------------------#

st.set_page_config(page_title="The model", page_icon='🕰️',
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

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

Listd = ["total","lineaA","lineaB","lineaC","lineaD","lineaE","lineaH" ]

def bokehlineplot2(List):
  
  p = figure(width=800, height=250, x_axis_type="datetime")
  p.title.text = 'Click on legend entries to mute the corresponding lines'
  for data, name, color in zip([Practica,PracticaA,PracticaB,PracticaC,PracticaD,PracticaE,PracticaH],List, Spectral7):
    df = pd.DataFrame(data)
    df['fecha'] = pd.to_datetime(df['fecha'])
    p.line(df['fecha'], df['pax_total'], line_width=2, color=color, alpha=0.8,
                 muted_color=color, muted_alpha=0.2, legend_label=name)
  p.legend.location = "top_left"
  p.legend.click_policy="mute"
  return p


p = bokehlineplot2(Listd)
container = st.container()
container.bokeh_chart(p,use_container_width = True)

#------------------------ The model------------------------#

#-------Pickle----#

with open("data/model_fb.pkl", 'rb') as Prophet_model_fb:
        model_fb = pickle.load(Prophet_model_fb)
    
future_pd = model_fb.make_future_dataframe(
    periods = 42,
    freq = 'm',
    include_history=True
)

predictions_fb = model_fb.predict(future_pd)

predict_fig = model_fb.plot(predictions_fb, \
                            xlabel='Month', \
                            ylabel='total', )

st.pyplot(predict_fig, clear_figure = True)

fig2 = model_fb.plot_components(predictions_fb)

st.pyplot(fig2, clear_figure = True)

# predict over the dataset
predictions_fb = model_fb.predict(future_pd)


#---grafico genial----# Esta aca para intentar hacer otra al final de modelo con los errores!
def predictgrapht():
    fig = plot_plotly(model_fb,predictions_fb,
            ylabel='total',
            changepoints=False,
            trend=True,
            uncertainty=True,
        )

    #Load data
    df = predictions_fb

    # Create figure

    fig.add_trace(
        go.Scatter(x=list(df.ds), y=list(df.trend)))
    #fig.add_trace(
        #go.Scatter(x=list(df.ds), y=list(df.yhat)))

    # Set title
    fig.update_layout(
        title_text="Time series with range slider and selectors"
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return
  

# evaluate = st.sidebar.checkbox(
#   "Evaluate my model", value=True)
