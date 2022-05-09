

import streamlit as st
import altair as alt
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.tools as tls
import time
import datetime
import os
import glob
import shutil
import sys


# ######################
# create a progress bar for the app
# ######################
progress_bar = st.sidebar.progress(0)  # initialize the progress bar
# initialize the status text area for the progress bar
status_text = st.sidebar.empty()
# create a random number matrix (5 rows, 4 columns) gausian distribution with mean 0 and standard deviation 1
last_row = np.random.randn(5, 4)
# create a binomial distribution n is the number of trials, p is the probability of success and size is the number of samples
binom_dist = np.random.binomial(n=10, p=0.5, size=1000)

st.write("plotting random normal distribution")
chart = st.line_chart(last_row, width=600, height=400)  # create a line chart

st.write("plotting random binomial distribution")
chart2 = st.line_chart(binom_dist)  # create a line chart
# sc

for i in range(100):
    # update the progress bar with the current value
    progress_bar.progress(i)
    # update the status text area with the current value
    status_text.text(f"loading {i}%")
    # update the line chart with the new data
    new_row = np.random.randn(5, 4)
    chart.add_rows(new_row)

    last_row = np.vstack((last_row, new_row))
    # sleep for a bit so the progress bar can be seen
    time.sleep(0.1)

st.write("plotting normal distribution histogram")
df = pd.DataFrame(last_row).reset_index(inplace=False)
st.write(df)
# create a figure and axes object for the histogram plot with a size of 15x12
fig, ax = plt.subplots(figsize=(15, 12))
# melt the dataframe to create a long format dataframe
df_long = pd.melt(df, value_vars=df.columns.tolist()[1:],
                  id_vars=["index"], var_name='variable', value_name='value')

st.write(df_long)
sns.set(style="whitegrid")
sns.displot(x="value", data=df_long, hue="variable", kind="kde", alpha=0.5,
            ax=ax)  # create a histogram plot , `kind` must be one of `bar`, `box`, `kde`,'ecdf', `hex`, `hist`, `pie` or `scatter`
st.pyplot()  # display the histogram plot

st.write(last_row)  # print the random number
st.write(binom_dist)  # print the random number
progress_bar.progress(100)  # update the progress bar


st.button("Start")  # create a button to start the progress bar
st.button("Re-run")  # create a button to re-run the progress bar
