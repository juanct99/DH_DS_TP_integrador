import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#------------------------ The model ------------------------#

st.set_page_config(page_title="The model", page_icon=None,
                   layout="wide", initial_sidebar_state="auto",
                   menu_items=None)
#evaluate = st.sidebar.checkbox(
#"Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"])

current_dir = os.getcwd()
path = os.path.join(current_dir, "data/dfs_day_grouped.csv")

@st.cache_data(show_spinner=True)
def read_file(path):
   df = pd.read_csv(path)
   return df

df = read_file(path)
df.fecha = pd.to_datetime(df.fecha, format='%Y-%m-%d')
df.hora = df.hora.astype(int)

grouped_by = {
   "Día": "D",
   "Semana": "W",
   "Mes": "MS"
}
tipo_dia_dict = {"H": "Habiles",
                 "F": "Feriados",
                 "S": "Sabados",
                 "D": "Domingos"}

with st.sidebar.expander("Lineas", expanded=False):
   lineas = df.linea.unique().tolist()
   linea = st.multiselect("Lineas incluidas",lineas,
                           default=lineas,
                           format_func=lambda x: x.replace('Linea', ''))
  

def bokehlineplot2():

  p = figure(width=800, height=250, x_axis_type="datetime")
  p.title.text = 'Click on legend entries to mute the corresponding lines'

  for data, name, color in zip([df_filtered], [df_filtered], Spectral4):
      df = pd.DataFrame(data)
      df['date'] = pd.to_datetime(df['date'])
      p.line(df['date'], df['close'], line_width=2, color=color, alpha=0.8,
             muted_color=color, muted_alpha=0.2, legend_label=name)

  p.legend.location = "top_left"
  p.legend.click_policy="mute"

  show(p)

return fig  

evaluate = st.sidebar.checkbox("Evaluate my model", value=True)
