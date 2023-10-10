import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import plotly.express as px


#------------------------ Objetives and description ------------------------#

st.set_page_config(page_title="Objetives", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded",
                   menu_items=None)

descripcion_proyecto = ("""
<div style="text-align: justify; line-height: 1.5; padding-bottom: 12px;">
  <span style="font-size: 17px;">
    <br>
    El conjunto de datos con el que estaremos trabajando consiste en la cantidad de pasajeros que utilizaron el sistema de transporte subterráneo de Buenos Aires en sus distintas líneas diariamente durante los años 2017 a 2022, lo que suma un total de aproximadamente <strong>60M de registros</strong>.
    <br>
    <br>
    Realizaremos tareas de limpieza, transformación y visualización de datos, con el objetivo de generar un escenario óptimo para la elección de modelos de series temporales. Evaluaremos la tendencia, estacionalidad y aleatoriedad de los datos.
    <br>
    <br>
    Por último, utilizaremos los modelos generados para predecir la cantidad de pasajeros que utilizarán cierta línea de subte en un día determinado.
  </span>
</div>
<br>
<br>
""")

objetivos = (f"""
    <br>
    <ul style="list-style-type: disc;">
      <li style="font-size: 17px;">Procesar, limpiar, transformar y estandarizar cada una de las fuentes de informacion.</li>
      <li style="font-size: 17px;">Crear visualizaciones claras para comprender la información disponible.</li>
      <li style="font-size: 17px;">Analizar patrones y tendencias en el uso del subterráneo de Buenos Aires a lo largo del tiempo.</li>
      <li style="font-size: 17px;">Identificar factores que influyen en la cantidad de pasajeros en momentos específicos.</li>
      <li style="font-size: 17px;">Generar diversos modelos de series temporales para comparar resultados.</li>
      <li style="font-size: 17px;">Evaluar tendencias, estacionalidad y aleatoriedad de los datos.</li>
      <li style="font-size: 17px;">Utilizar los modelos para predecir la cantidad de pasajeros en una línea de subte en un día específico.</li>
    </ul>
""")

informacion = (f"""
    <br>
    <ul style="list-style-type: disc;">
      <li style="font-size: 17px;">Procesar, pruebas necesarias para visualizar lo requerido en Git.</li>
      """)

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/bocas-de-subte.csv")

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

bocas_del_subte = read_file(path)

def mapa_de_bocas():
        geo_subte = gpd.GeoDataFrame(bocas_del_subte,
                                geometry = gpd.points_from_xy(bocas_del_subte.long, bocas_del_subte.lat),
                                crs=3857)
        geo_subte.rename(columns={"long": "lon",
                                "linea": "Linea",
                                "numero_de_": 'Molinetes'}, inplace=True)

        colores = {'A': 'lightblue','B': 'red',
                'C': 'blue','D': 'green',
                'E': 'purple','H': 'yellow'}

        fig = px.scatter_mapbox(geo_subte, lat='lat', lon='lon',
                                hover_name='Linea', zoom=11,
                                mapbox_style='carto-positron',
                                color='Linea', color_discrete_map=colores,
                                size='Molinetes', size_max=10,
                                width=500, height=500,
                                center={'lat': -34.60205, 'lon': -58.43135})
        fig.update_layout(font_color='black',
                        hoverlabel_bordercolor='black',
                        hoverlabel_bgcolor = 'white',
                        legend_bgcolor="#FDFFCD",
                        legend_borderwidth=1)

        return fig

st.header("🗒️Breve description")
st.write(descripcion_proyecto, unsafe_allow_html=True)


st.header("🎯Objetivos")
c1, c2 = st.columns(2)
with c1:
        st.write(objetivos, unsafe_allow_html=True)
with c2:
        st.plotly_chart(mapa_de_bocas(), use_container_width=True)
