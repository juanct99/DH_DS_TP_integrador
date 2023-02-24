import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

#------------------------ The model ------------------------#

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

st.dataframe(df.head())


# evaluate = st.sidebar.checkbox(
#   "Evaluate my model", value=True)


# def bokehlineplot2():

#   p = figure(width=800, height=250, x_axis_type="datetime")
#   p.title.text = 'Click on legend entries to mute the corresponding lines'

#   for data, name, color in zip([df_filtered], [df_filtered], Spectral4):
#       df = pd.DataFrame(data)
#       df['date'] = pd.to_datetime(df['date'])
#       p.line(df['date'], df['close'], line_width=2, color=color, alpha=0.8,
#              muted_color=color, muted_alpha=0.2, legend_label=name)

#   p.legend.location = "top_left"
#   p.legend.click_policy="mute"

#   show(p)
#   return fig  
