from turtle import color
import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import plotly.express as px


#------------------------ Objetives and description ------------------------#

st.set_page_config(page_title="Objetives", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded",
                   menu_items=None)


st.write("""<div style="text-align: left"><span style="font-size: 26px">
        🎯 Objetivos y descripción del proyecto
         </span></div>""",
         unsafe_allow_html=True)

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

st.plotly_chart(mapa_de_bocas())

