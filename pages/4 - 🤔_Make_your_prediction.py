import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime as dt


import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot

#------------------------ Make your prediction ------------------------#

st.set_page_config(page_title="Predict", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)

today = dt.date.today()
dias_es = {"Monday": "Lunes",
           "Tuesday": "Martes",
           "Wednesday": "Miércoles",
           "Thursday": "Jueves",
           "Friday": "Viernes",
           "Saturday": "Sábado",
           "Sunday": "Domingo"}
tipo_dia = {"Lunes": "H",
            "Martes": "H",
            "Miércoles": "H",
            "Jueves": "H",
            "Viernes": "H",
            "Sábado": "S",
            "Domingo": "D"}
estaciones_y_lineas = pickle.load(open("data/estaciones_y_lineas.pickle", "rb"))


c1,c2 = st.columns([1,1])
with c1:
    fecha = st.date_input("Date", value=None , min_value=None , max_value=None , key=None )
    is_feriado = st.checkbox("Es feriado", value=False, key=None)
with c2:
    hora = st.selectbox('Hora',
                        (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23))

c3,c4 = st.columns([1,1])
with c3:
    linea = st.selectbox('Linea',
                        estaciones_y_lineas.linea.unique().tolist())
with c4:
    estacion = estaciones_y_lineas[estaciones_y_lineas.linea == linea].estacion.sort_values().tolist()
    estacion = st.selectbox("Estación",estacion)

dia_de_la_semana = dias_es.get(fecha.strftime("%A"))
tipo_dia = "F" if is_feriado else tipo_dia.get(dia_de_la_semana)

variables = {
    "fecha": str(fecha),
    "hora": hora,
    "linea": linea,
    "estacion": estacion,
    "dia_de_la_semana": dia_de_la_semana,
    "tipo_dia": tipo_dia
}

st.write(variables)


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

# predict over the dataset
predictions_fb = model_fb.predict(future_pd)


#---grafico genial----#
def predictgrapht(modelo,fcst):
    fig = plot_plotly(modelo,fcst,
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

    fig.show()
    return fig

container = st.container()
container.plotly_chart(predictgrapht(model_fb,predictions_fb), use_container_width=True, sharing="streamlit", theme="streamlit")

 
