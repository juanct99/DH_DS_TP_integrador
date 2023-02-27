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
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot

#------------------------ Make your prediction ------------------------#

st.set_page_config(page_title="Predict", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)


current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

df = read_file(path)
df.fecha = pd.to_datetime(df.fecha, format="%Y-%m-%d")

horas = df.hora.astype(int).sort_values().unique().tolist()
sentido = df.sentido.unique().tolist()

today = dt.date.today()
sentidos = {"N": "Norte",
    "S": "Sur",
    "E": "Este",
    "O": "Oeste",
}
estaciones_y_lineas = pickle.load(open("data/estaciones_y_lineas.pickle", "rb"))


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
    sentido_mask = (df.sentido == sentidos.get(sentido)) | (df.sentido == "-")
    df_filtrado = df[hour_mask & linea_mask & estaciones_mask & sentido_mask]
            
    return fecha, boton, df_filtrado

fecha_prediccion, boton, df_filtrado = sidebar_form()

if boton:
    last_date = df_filtrado.fecha.max().date()
    cantidad_dias = (fecha_prediccion - last_date).days


else:
    st.info("Configura las variables de entrada y haz click en 'Hacer predicción' para ver los resultados")

#------------------------ The model------------------------#


# with open("data/model_fb.pkl", 'rb') as Prophet_model_fb:
#         model_fb = pickle.load(Prophet_model_fb)
    
# future_pd = model_fb.make_future_dataframe(
#     periods = 42,
#     freq = 'm',
#     include_history=True
# )

# predictions_fb = model_fb.predict(future_pd)

# # predict over the dataset
# predictions_fb = model_fb.predict(future_pd)

# #---grafico genial----#
# def predictgrapht(modelo,fcst):
#     fig = plot_plotly(modelo,fcst,
#             ylabel='total',
#             changepoints=False,
#             trend=True,
#             uncertainty=True,
#         )

#     #Load data
#     df = predictions_fb

#     # Create figure

#     fig.add_trace(
#         go.Scatter(x=list(df.ds), y=list(df.trend)))
#     #fig.add_trace(
#         #go.Scatter(x=list(df.ds), y=list(df.yhat)))

#     # Set title
#     fig.update_layout(
#         title_text="Time series with range slider and selectors"
#     )

#     # Add range slider
#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1,
#                          label="1m",
#                          step="month",
#                          stepmode="backward"),
#                     dict(count=6,
#                          label="6m",
#                          step="month",
#                          stepmode="backward"),
#                     dict(count=1,
#                          label="YTD",
#                          step="year",
#                          stepmode="todate"),
#                     dict(count=1,
#                          label="1y",
#                          step="year",
#                          stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         )
#     )

#     return fig

# container = st.container()
# container.plotly_chart(predictgrapht(model_fb,predictions_fb), use_container_width=True, sharing="streamlit", theme="streamlit")