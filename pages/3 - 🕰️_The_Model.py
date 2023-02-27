import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np

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
from prophet.plot import plot_plotly,get_forecast_component_plotly_props, plot_components_plotly, seasonality_plot_df, get_seasonality_plotly_props
from prophet.plot import add_changepoints_to_plot

#------------------------ The model Intro ------------------------#

st.set_page_config(page_title="The model", page_icon='🕰️',
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

@st.cache_data(show_spinner=False)
def read_file(path):
   df = pd.read_csv(path)
   return df

with st.spinner("Cargando datos..."):
    df = read_file(path)

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
color = "#FDFFCD"
def bokehlineplot2(List,color=color):

    p = figure(width=800, height=250, x_axis_type="datetime",background_fill_color=color, outline_line_color="black",
            border_fill_color = color)
    for data, name, color in zip([Practica,PracticaA,PracticaB,PracticaC,PracticaD,PracticaE,PracticaH],List, Spectral7):
        df = pd.DataFrame(data)
        df['fecha'] = pd.to_datetime(df['fecha'])
        p.line(df['fecha'], df['pax_total'], line_width=2, color=color, alpha=0.8,
            muted_color=color, muted_alpha=0.2, legend_label=name)
    p.legend.location = "top_left"
    p.legend.click_policy="mute"
    p.legend.background_fill_color = "#FEFFE9"
    return p


p = bokehlineplot2(Listd)
container = st.container()
container.bokeh_chart(p,use_container_width = True)

#------------------------ The model------------------------#

with open("data/model_fb.pkl", 'rb') as Prophet_model_fb:
        model = pickle.load(Prophet_model_fb)
    
future_pd = model.make_future_dataframe(
    periods = 42,
    freq = 'm',
    include_history=True
)

forecast = model.predict(future_pd)

def plotly_prediction(m, fcst, uncertainty=True, plot_cap=True, trend=True, changepoints=False,
                changepoints_threshold=0.01, xlabel='ds', ylabel='y', figsize=(900, 600)):
    
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    actual_color = 'black'
    cap_color = 'black'
    trend_color = '#B23B00'
    line_width = 2
    marker_size = 4

    data = []
    # Add actual
    data.append(go.Scatter(
        name='Actual',
        x=m.history['ds'],
        y=m.history['y'],
        marker=dict(color=actual_color, size=marker_size),
        mode='markers'
    ))
    # Add lower bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip'
        ))
    # Add prediction
    data.append(go.Scatter(
        name='Predicted',
        x=fcst['ds'],
        y=fcst['yhat'],
        mode='lines',
        line=dict(color=prediction_color, width=line_width),
        fillcolor=error_color,
        fill='tonexty' if uncertainty and m.uncertainty_samples else 'none'
    ))
    # Add upper bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            fillcolor=error_color,
            fill='tonexty',
            hoverinfo='skip'
        ))
    # Add caps
    if 'cap' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    # Add trend
    if trend:
        data.append(go.Scatter(
            name='Trend',
            x=fcst['ds'],
            y=fcst['trend'],
            mode='lines',
            line=dict(color=trend_color, width=line_width),
        ))
    # Add changepoints
    if changepoints and len(m.changepoints) > 0:
        signif_changepoints = m.changepoints[
            np.abs(np.nanmean(m.params['delta'], axis=0)) >= changepoints_threshold
        ]
        data.append(go.Scatter(
            x=signif_changepoints,
            y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
            marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                        line=dict(width=line_width)),
            mode='markers',
            hoverinfo='skip'
        ))

    layout = dict(
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        yaxis=dict(
            title=ylabel
        ),
        xaxis=dict(
            title=xlabel,
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    
    return fig

st.write("""<p style="text-align: justify;">Esta visualización muestra varios datos:</p>
<ul>
  <li>La línea azul muestra las predicciones realizadas por el modelo en los períodos de entrenamiento y validación.</li>
  <li>La sombra azul alrededor es un intervalo de incertidumbre del 80% en estas predicciones.</li>
  <li>Los puntos negros son los valores reales del objetivo en el período de entrenamiento.</li>
  <li>La línea roja es la tendencia estimada por el modelo, y las líneas verticales muestran los puntos de cambio en los que evoluciona esta tendencia.</li>
</ul>"""
,unsafe_allow_html=True)

st.info("Puede usar el control deslizante en la parte inferior o los botones en la parte superior para enfocarse en un período de tiempo específico")
st.plotly_chart(plotly_prediction(model,forecast), use_container_width = True)

st.write("""<p>La previsión que genera Prophet es la suma de diferentes aportaciones:</p>
<ul>
<li>Tendencia</li>
<li>Estacionalidades</li>
<li>Otros factores como vacaciones o regresores externos</li>
</ul>
<p>Las siguientes visualizaciones muestran este desglose y le permite comprender cómo contribuye cada componente al valor final pronosticado por el modelo.</p>
<style>
p, ul {
  text-align: justify;
}
</style>""",unsafe_allow_html=True)


st.plotly_chart(plot_components_plotly(model, forecast, figsize=(900,300)), use_container_width=True)

# predict over the dataset
predictions_fb = model.predict(future_pd)


#---grafico genial----# Esta aca para intentar hacer otra al final de modelo con los errores!
def predictgrapht():
    fig = plot_plotly(model,forecast,
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
    return fig
