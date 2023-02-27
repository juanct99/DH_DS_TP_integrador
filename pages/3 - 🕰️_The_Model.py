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
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot

#------------------------ The model Intro ------------------------#

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

def plotly_prediction(m, fcst, uncertainty=True, plot_cap=True, trend=False, changepoints=False,
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
  <li>Puede usar el control deslizante en la parte inferior o los botones en la parte superior para enfocarse en un período de tiempo específico.</li>
</ul>"""
,unsafe_allow_html=True)

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


def get_forecast_component_plotly_props(m, fcst, name, uncertainty=True, plot_cap=False):
    
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    cap_color = 'black'
    zeroline_color = '#AAA'
    line_width = 2

    range_margin = (fcst['ds'].max() - fcst['ds'].min()) * 0.05
    range_x = [fcst['ds'].min() - range_margin, fcst['ds'].max() + range_margin]

    text = None
    mode = 'lines'
    if name == 'holidays':
        
        # Combine holidays into one hover text
        holidays = m.construct_holiday_dataframe(fcst['ds'])
        holiday_features, _, _ = m.make_holiday_features(fcst['ds'], holidays)
        holiday_features.columns = holiday_features.columns.str.replace('_delim_', '', regex=False)
        holiday_features.columns = holiday_features.columns.str.replace('+0', '', regex=False)
        text = pd.Series(data='', index=holiday_features.index)
        for holiday_feature, idxs in holiday_features.iteritems():
            text[idxs.astype(bool) & (text != '')] += '<br>'  # Add newline if additional holiday
            text[idxs.astype(bool)] += holiday_feature

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=fcst['ds'],
        y=fcst[name],
        mode=mode,
        line=go.scatter.Line(color=prediction_color, width=line_width),
        text=text,
    ))
    if uncertainty and m.uncertainty_samples and (fcst[name + '_upper'] != fcst[name + '_lower']).any():
        if mode == 'markers':
            traces[0].update(
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=fcst[name + '_upper'],
                    arrayminus=fcst[name + '_lower'],
                    width=0,
                    color=error_color
                )
            )
        else:
            traces.append(go.Scatter(
                name=name + '_upper',
                x=fcst['ds'],
                y=fcst[name + '_upper'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color)
            ))
            traces.append(go.Scatter(
                name=name + '_lower',
                x=fcst['ds'],
                y=fcst[name + '_lower'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color),
                fillcolor=error_color,
                fill='tonexty'
            ))
    if 'cap' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))

    xaxis = go.layout.XAxis(
        type='date',
        range=range_x)
    yaxis = go.layout.YAxis(rangemode='normal' if name == 'trend' else 'tozero',
                            title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    if name in m.component_modes['multiplicative']:
        yaxis.update(tickformat='%', hoverformat='.2%')
    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis}

def plot_components_plotly(
        m, fcst, uncertainty=True, plot_cap=True, figsize=(900, 300)):

    # Identify components to plot and get their Plotly props
    components = {}
    components['trend'] = get_forecast_component_plotly_props(
        m, fcst, 'trend', uncertainty, plot_cap)
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components['holidays'] = get_forecast_component_plotly_props(
            m, fcst, 'holidays', uncertainty)

    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components['extra_regressors_{}'.format(mode)] = get_forecast_component_plotly_props(
                m, fcst, 'extra_regressors_{}'.format(mode))
    for seasonality in m.seasonalities:
        components[seasonality] = get_seasonality_plotly_props(m, seasonality)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(rows=len(components), cols=1, print_grid=False)
    fig['layout'].update(go.Layout(
        showlegend=False,
        width=figsize[0],
        height=figsize[1] * len(components)
    ))
    for i, name in enumerate(components):
        if i == 0:
            xaxis = fig['layout']['xaxis']
            yaxis = fig['layout']['yaxis']
        else:
            xaxis = fig['layout']['xaxis{}'.format(i + 1)]
            yaxis = fig['layout']['yaxis{}'.format(i + 1)]
        xaxis.update(components[name]['xaxis'])
        yaxis.update(components[name]['yaxis'])
        for trace in components[name]['traces']:
            fig.append_trace(trace, i + 1, 1)
    return fig

def seasonality_plot_df(m, ds):
    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props['condition_name'] is not None:
            df_dict[props['condition_name']] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df

def get_seasonality_plotly_props(m, name, uncertainty=True):
    """Prepares a dictionary for plotting the selected seasonality with Plotly

    Parameters
    ----------
    m: Prophet model.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.

    Returns
    -------
    A dictionary with Plotly traces, xaxis and yaxis
    """
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    line_width = 2
    zeroline_color = '#AAA'

    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    if (m.history['ds'].dt.hour == 0).all():  # Day Precision
        plot_points = np.floor(period).astype(int)
    elif (m.history['ds'].dt.minute == 0).all():  # Hour Precision
        plot_points = np.floor(period * 24).astype(int)
    else:  # Minute Precision
        plot_points = np.floor(period * 24 * 60).astype(int)
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points, endpoint=False))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=df_y['ds'],
        y=seas[name],
        mode='lines',
        line=go.scatter.Line(color=prediction_color, width=line_width)
    ))
    if uncertainty and m.uncertainty_samples and (seas[name + '_upper'] != seas[name + '_lower']).any():
        traces.append(go.Scatter(
            name=name + '_upper',
            x=df_y['ds'],
            y=seas[name + '_upper'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color)
        ))
        traces.append(go.Scatter(
            name=name + '_lower',
            x=df_y['ds'],
            y=seas[name + '_lower'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color),
            fillcolor=error_color,
            fill='tonexty'
        ))

    # Set tick formats (examples are based on 2017-01-06 21:15)
    if period <= 2:
        tickformat = '%H:%M'  # "21:15"
    elif period < 7:
        tickformat = '%A %H:%M'  # "Friday 21:15"
    elif period < 14:
        tickformat = '%A'  # "Friday"
    else:
        tickformat = '%B %e'  # "January  6"

    range_margin = (df_y['ds'].max() - df_y['ds'].min()) * 0.05
    xaxis = go.layout.XAxis(
        tickformat=tickformat,
        type='date',
        range=[df_y['ds'].min() - range_margin, df_y['ds'].max() + range_margin]
    )

    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        yaxis.update(tickformat='%', hoverformat='.2%')

    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis}

st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

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
    return
