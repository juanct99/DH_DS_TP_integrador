import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime as dt
import pandas as pd
import os


import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot

#------------------------ Make your prediction ------------------------#

st.set_page_config(page_title="Predict", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)


current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")
feriados = os.path.join(current_dir, "data/feriados.csv")
today = dt.date.today()

@st.cache_data(show_spinner=True)
def read_file(path, sep=","):
   df = pd.read_csv(path, sep=sep)
   return df

df = read_file(path, sep=",")
df.fecha = pd.to_datetime(df.fecha, format="%Y-%m-%d")

horas = df.hora.astype(int).sort_values().unique().tolist()
sentidos = {"N": "Norte","S": "Sur","E": "Este","O": "Oeste", "-": "Sin sentido específico"}
estaciones_y_lineas = pickle.load(open("data/estaciones_y_lineas.pickle", "rb"))

def getFeriados(path_feriados, sep):  
    feriados = read_file(path_feriados, sep=sep)
    feriados.fecha_feriado = pd.to_datetime(feriados.fecha_feriado, dayfirst=True)
    feriados = feriados.rename(columns={'fecha_feriado': 'ds'})
    feriados["holiday"] = 'feriado'
    feriados = feriados[['ds', 'holiday']]
    return feriados
feriados = getFeriados(feriados, sep = ";")

st.header("🚇Predice que tan concurrido estará el subte al momento de tu viaje")

def sidebar_form():
    st.sidebar.subheader("Variables de entrada")
    fecha = st.sidebar.date_input("Fecha")
    hora = st.sidebar.selectbox('Hora',horas, index=8)
    linea = st.sidebar.selectbox('Linea',estaciones_y_lineas.linea.unique().tolist())
    estacion = estaciones_y_lineas[estaciones_y_lineas.linea == linea].estacion.sort_values().tolist()
    estacion = st.sidebar.selectbox("Estación",estacion)
    sentidos_posibles = df[(df.estacion == estacion)].sentido.unique().tolist()
    sentido = st.sidebar.selectbox('Sentido',sentidos_posibles, format_func=lambda x: sentidos.get(x))
    
    st.sidebar.write("")
    boton = st.sidebar.button("Hacer predicción", use_container_width=True, type="primary")

    hour_mask = df.hora == hora
    linea_mask = df.linea == linea
    estaciones_mask = df.estacion == estacion
    sentido_mask = (df.sentido == sentido) | (df.sentido == "-")
    df_filtrado = df[hour_mask & linea_mask & estaciones_mask & sentido_mask]
            
    return fecha, boton, df_filtrado

fecha_prediccion, boton, df_filtrado = sidebar_form()

def predict(df_filtrado, fecha_prediccion):

    model = Prophet(holidays=feriados)

    df_train = df_filtrado[['fecha','pax_total']].copy()
    df_train.columns = ["ds", "y"]
    
    model.fit(df_train)

    last_date = df_filtrado.fecha.max().date()
    dias = int((fecha_prediccion - last_date).days*1.5)
    
    
    df_future_prediction = model.make_future_dataframe(
        periods = dias,
        freq = 'D',
        include_history=True
    )

    prediction = model.predict(df_future_prediction)

    return model, prediction

if boton:
    
    try:
        model, prediction = predict(df_filtrado, fecha_prediccion)
        st.success("Predicción realizada con éxito")
        st.dataframe(prediction.head())
        
        st.plotly_chart(plot_plotly(model, prediction, trend = True), use_container_width=True)
        st.plotly_chart(plot_components_plotly(model, prediction), use_container_width=True)
        
        
    except Exception as e:
        st.exception(f"Error: {e}")
    
else:
    st.info("Configura las variables de entrada y haz click en 'Hacer predicción' para ver los resultados")