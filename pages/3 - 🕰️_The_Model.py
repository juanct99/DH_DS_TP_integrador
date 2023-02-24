import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#------------------------ The model ------------------------#

st.set_page_config(page_title="The model", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)
evaluate = st.sidebar.checkbox(
    "Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"]
)
