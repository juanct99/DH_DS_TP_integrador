import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, VArea, HoverTool
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


def agrupacion(dfinput):
    
    sumatoria = dfinput.groupby(by=['fecha','linea','tipo_dia'])['pax_total'].sum().reset_index()

    sumatoria['fecha'] = pd.to_datetime(sumatoria['fecha'], dayfirst=True)   

    data = sumatoria.set_index('fecha')

    data.sort_values(by='fecha',ascending=False)

    y = data['pax_total'].resample('M').sum()

    y_output = pd.DataFrame({'pax_total': y}).reset_index()

    y_output.index = pd.PeriodIndex(y_output['fecha'], freq='M')
        
    return y_output

total_y_lineas = ["Total"] + df.linea.unique().tolist()

dfs_filtrados_y_agrupados = [agrupacion(df[df.linea == linea]) for linea in df.linea.unique().tolist()]
dfs_filtrados_y_agrupados.insert(0,agrupacion(df))

st.header("⏱️Series de tiempo")
st.write("")

texto_intro = """
En esta sección se presenta el modelo de predicción de usuarios del subterraneo de la ciudad de Buenos Aires.\n
Cabe destacar que el modelo es de periodicidad diaria para la totalidad de las lineas y horarios de la red, la posibilidad de realizar 
predicciones para una linea, horario, estacion y sentido en particular se encuentra en la sección de 'Hacé tu predicción'. \n
A pesar de esto, la metodologia de aprendizaje y entrenamiento del modelo es la misma en ambos escenarios.\n
"""

st.write(texto_intro)


color = "#FDFFCD"
def bokehlineplot(legends_names,back_color = color):

    p = figure(width=800, height=250, x_axis_type="datetime",background_fill_color=back_color, outline_line_color="black",
            border_fill_color = back_color)
    
    for data, name, color in zip(dfs_filtrados_y_agrupados,legends_names, Spectral7):
        df = pd.DataFrame(data)
        df['fecha'] = pd.to_datetime(df['fecha'])
        p.line(df['fecha'], df['pax_total'], line_width=2, color=color, alpha=0.8,
            muted_color=color, muted_alpha=0.2, legend_label=name, name='name')
        
        hover = HoverTool(tooltips=[('Fecha', '@x{%F}'), ('Pax Total', '@y{0,0}')], formatters={'@x': 'datetime'}, mode='mouse')
        p.add_tools(hover)
    
    p.legend.location = "top_left"
    p.legend.click_policy="mute"
    p.legend.background_fill_color = "#FEFFE9"
    return p


st.subheader("Uso por linea de subte")
st.info("Para quitar una linea en particular, haga click en la leyenda")

p = bokehlineplot(total_y_lineas)
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