import streamlit as st
from PIL import Image
import time

#------------------------ Homepage ------------------------#

st.set_page_config(page_title="Home", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded",
                   menu_items=None)

col1, col2 = st.columns([1,3])
with col2:
    st.header(f"Data Science - Trabajo Final Integrador")

with col1:
    dh_logo = Image.open("images/dh_logo.png")
    st.image(dh_logo, width=110, use_column_width=False)
    
st.write("""<div style="text-align: center"><span style="font-size: 20px">
         Series de tiempo aplicado a la predicción de cantidad de usarios
         del subteráneo de la Ciudad de Buenos Aires mediante el procesamiento de datos historicos desde 2017 a 2022.
         </span></div>""",
         unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")

c1, c2, c3, c4, c5, c6 = st.columns([3,1,1,1,1,3])
with c1:
    st.subheader("Integrantes")
    st.write("- _Mateo Zarza_")
    st.write("- _Luis Carrero_")
    st.write("- _Matias Arias_")
    st.write("- [_Juan Cruz Traverso_](https://www.linkedin.com/in/jcruztraverso/)")

    st.write('<div style="color: rgba(128, 128, 128, 0.5); padding: 10px">Última actualización: 2023-02</div>', unsafe_allow_html=True)

with c6:
    subte = Image.open("images/subte.png")
    st.image(subte, width=300, use_column_width=False)
