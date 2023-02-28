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

@st.cache_data(show_spinner=False)
def read_file(path, sep=","):
   df = pd.read_csv(path, sep=sep)
   return df

with st.spinner('Cargando datos...'):
    df = read_file(path, sep=",")
    
df.fecha = pd.to_datetime(df.fecha, format="%Y-%m-%d")

horas = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
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

st.sidebar.subheader("Variables de entrada")
def sidebar_form():
    fecha = st.sidebar.date_input("Fecha", key="date_input")
    hora = st.sidebar.selectbox('Hora',horas, index=3)
    linea = st.sidebar.selectbox('Linea',estaciones_y_lineas.linea.unique().tolist())
    estacion = estaciones_y_lineas[estaciones_y_lineas.linea == linea].estacion.sort_values().tolist()
    estacion = st.sidebar.selectbox("Estación",estacion)
    
    st.sidebar.write("")
    boton = st.sidebar.button("Hacer predicción", use_container_width=True, type="primary")
    hour_mask = df.hora == hora
    hour_mask_plus_1 = df.hora == hora +1 if hora != 23 else df.hora == 0
    hour_mask_minus_1 = df.hora == hora -1 if hora != 0 else df.hora == 23
    linea_mask = df.linea == linea
    estaciones_mask = df.estacion == estacion
    
    df_filtrado = df[hour_mask & linea_mask & estaciones_mask ]
    df_filtrado_plus_1 = df[hour_mask_plus_1 & linea_mask & estaciones_mask]
    df_filtrado_minus_1 = df[hour_mask_minus_1 & linea_mask & estaciones_mask]
            
    return fecha, hora, boton, df_filtrado, df_filtrado_plus_1, df_filtrado_minus_1

fecha_prediccion, hour, boton, df_filtrado, df_filtrado_plus_1, df_filtrado_minus_1 = sidebar_form()

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
    with st.spinner('Realizando predicciones...'):
        try:
            model, prediction = predict(df_filtrado, fecha_prediccion)
            model_plus_1, prediction_plus_1 = predict(df_filtrado_plus_1, fecha_prediccion)
            model_minus_1, prediction_minus_1 = predict(df_filtrado_minus_1, fecha_prediccion)
            
            st.success("Predicciones realizadas con éxito")
            st.write("")
            
            def pasajeros_metrics(prediccion):
                center = int(prediccion[prediccion.ds == str(fecha_prediccion)].yhat.values[0]) if int(prediccion[prediccion.ds == str(fecha_prediccion)].yhat.values[0]) > 0 else 0
                limite_superior = int(prediccion[prediccion.ds == str(fecha_prediccion)].yhat_upper.values[0])
                limite_inferior = int(prediccion[prediccion.ds == str(fecha_prediccion)].yhat_lower.values[0]) if int(prediccion[prediccion.ds == str(fecha_prediccion)].yhat_lower.values[0]) > 0 else 0

                return limite_inferior, center, limite_superior
            
            inf, center, sup = pasajeros_metrics(prediction)
            inf_plus_1, center_plus_1, sup_plus_1 = pasajeros_metrics(prediction_plus_1)
            inf_minus_1, center_minus_1, sup_minus_1 = pasajeros_metrics(prediction_minus_1)
            
            def card_text(card_text, color):
                card = f"""
                <div style="border-radius: 10px; background-color: {color}; padding: 20px; display: inline-block;">
                <h3 style="font-size: 30px; text-align: center;">{card_text}</h3>
                </div>
                """
                return card
            
            def tarjetas(inf, center, sup):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(card_text(inf, "#D9FFDF"), unsafe_allow_html=True)
                with c2:
                    st.write(card_text(center,"#FFFECE"), unsafe_allow_html=True)
                with c3:    
                    st.write(card_text(sup,"#FFD1D0"), unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader(":green[Límite inferior]")
            with c2:
                st.subheader(":black[Predicción]")
            with c3:
                st.subheader(":red[Límite superior]")
            
            st.write("")
            st.subheader(f"{hour-1}:00 hs")
            tarjetas(inf_minus_1, center_minus_1, sup_minus_1)
            st.write("")
            st.write("")
            st.write("")
            
            st.write("")
            st.subheader(f"Hora seleccionada, {hour}:00 hs")
            tarjetas(inf, center, sup)
            st.write("")
            st.write("")
            st.write("")
            
            st.write("")
            st.subheader(f"{hour+1}:00 hs")
            tarjetas(inf_plus_1, center_plus_1, sup_plus_1)
            st.write("")
            st.write("")
            st.write("")
            
        except Exception as e:
            st.exception(f"Error: {e}")

    
else:
    st.info("Configura las variables de entrada y haz click en 'Hacer predicción' para ver los resultados")