

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
# initialize on sidebar the progress bar,0 is the starting point
progress_bar = st.sidebar.progress(0)
# initialize on sidebar the status text area,empty() is function to create an empty text area
status_text = st.sidebar.empty()

# ######################
# create a sidebar with buttons
# ######################
st.sidebar.header('Choose an option')

# create combo box for  distrubtion type
distribution_type = st.sidebar.selectbox('Select the distribution type', ['Normal', 'Binomial'],
                                         index=0, key='distribution_type')  # set the default value of the combo box to the first item in the list



if distribution_type == 'Normal':
    # create number input for the mean of the normal distribution
    mean_normal = st.sidebar.number_input(
        label='Enter the mean of the normal distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the standard deviation of the normal distribution
    std_normal = st.sidebar.number_input(
        label='Enter the standard deviation of the normal distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create normal distribution datasets with user inputs
    dist_normal = np.random.normal(mean_normal, std_normal, num_samples)

      # create a random number matrix (5 rows, 4 columns) gausian distribution with mean 0 and standard deviation 1
    last_row = np.random.randn(5, 4)

    st.write("plotting random normal distribution")
    # create a line chart width=600, height=400
    chart = st.line_chart(last_row, width=600, height=400)
    
    for i in range(101):
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
    st.hi
    # create a figure and axes object for the histogram plot with a size of 15x12
    fig, ax = plt.subplots(figsize=(15, 12))
    # melt the dataframe to create a long format dataframe
    df_long = pd.melt(df, value_vars=df.columns.tolist()[1:],
                      id_vars=["index"], var_name='variable', value_name='value')

    st.write(df_long)
    sns.set(style="whitegrid")
    sns.displot(x="value", data=df_long, hue="variable", kind="hist", alpha=0.5,
                ax=ax)  # create a histogram plot , `kind` must be one of `bar`, `box`, `kde`,'ecdf', `hex`, `hist`, `pie` or `scatter`
    st.pyplot()  # display the histogram plot

    st.Dataframe(last_row)  # print the random number



elif distribution_type == 'Binomial':
    # create number input for the number of trials
    num_trials = st.sidebar.number_input(
        label='Enter the number of trials', min_value=0, max_value=1000000, value=1000)
    # create number input for the probability of success
    prob_success = st.sidebar.number_input(
        label='Enter the probability of success', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create binomial distribution datasets with user inputs
    # create a binomial distribution n is the number of trials, p is the probability of success and size is the number of samples
    dist_binomial = np.random.binomial(n=num_trials, p=prob_success, size=num_samples)
    st.write("plotting random binomial distribution")
    chart2 = st.line_chart(dist_binomial)  # create a line chart
    
    st.Dataframe(dist_binomial)  # print the random number


else:
    st.write("Please select a distribution type")


# st.button("Start")  # create a button to start the progress bar
st.button("Re-run")  # create a button to re-run the progress bar
