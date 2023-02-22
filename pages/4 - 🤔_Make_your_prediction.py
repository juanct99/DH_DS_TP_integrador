import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime as dt

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
