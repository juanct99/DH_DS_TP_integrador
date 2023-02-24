import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#------------------------ The model ------------------------#

st.set_page_config(page_title="The model", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)
st.sidebar.title("4. Forecast")

# Choose whether or not to do future forecasts
make_future_forecast = st.sidebar.checkbox(
    "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
)
if make_future_forecast:
    with st.sidebar.expander("Horizon", expanded=False):
        dates = input_forecast_dates(df, dates, resampling, config, readme)
    with st.sidebar.expander("Regressors", expanded=False):
        datasets = input_future_regressors(
            datasets, dates, params, dimensions, load_options, date_col
        )
