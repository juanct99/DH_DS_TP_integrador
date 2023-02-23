import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure

#------------------------ Data Analisys ------------------------#

st.set_page_config(page_title="Data analisys", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)