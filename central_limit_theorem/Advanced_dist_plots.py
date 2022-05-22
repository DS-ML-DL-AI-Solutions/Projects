

from statistics import median
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
import scipy.stats as stats

# ######################
# create a sidebar with buttons
# ######################
st.sidebar.header('Choose an option')


# create combo box for  distrubtion type
distribution_type = st.sidebar.selectbox('Select the distribution type', ['All', 'Normal', 'Log Normal', 'Uniform', 'Triangular',
                                                                          'Exponential', 'Gamma', 'Beta', 'Poisson', 'Binomial', 'Chi-Squared', 'F-Distribution', 'Bernoulli', 'Negative Binomial', 'Geometric', 'Weibull'],
                                         index=0, key='distribution_type')  # set the default value of the combo box to the first item in the list

# create combo box for the user to select the type of graph
# create a list of options for the combo box
graph_type = st.sidebar.selectbox(
    'Select the type of graph', ['Bar Chart', 'Line Chart', 'Scatter Plot',
                                 'Histogram', 'Box Plot', 'Violin Plot', 'Density Plot'],
    index=3, key='graph_type')  # set the default value of the combo box to the first item in the list

plot_style = st.sidebar.selectbox(
    'Select the style of graph', [
        "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", 'presentation', 'xgridoff',
        'ygridoff', 'gridon', "none"],
    index=9, key='plot_style')  # set the default value of the combo box to the first item in the list

# set the default template to simple_white for the plotly charts
pio.templates.default = "simple_white"

# set the default template to ggplot2 for the plotly charts
px.defaults.template = plot_style
# set the default color scale to Blackbody for the plotly charts
px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
px.defaults.width = 900
px.defaults.height = 720

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

    # create a plot of the normal distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the normal distribution
        fig = px.bar(x=dist_normal, title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the normal distribution
        fig = px.line(y=dist_normal, title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the normal distribution
        fig = px.scatter(y=dist_normal, title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the normal distribution
        fig = px.histogram(x=dist_normal, title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the normal distribution
        fig = px.box(x=dist_normal, points='all', title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the normal distribution
        fig = px.violin(x=dist_normal, box=True, points='all',
                        title='Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the normal distribution
        fig = px.density_contour(y=dist_normal, title='Normal Distribution')
        st.plotly_chart(fig)

elif distribution_type == 'Log Normal':
    # create number input for the mean of the log normal distribution
    mean_log_normal = st.sidebar.number_input(
        label='Enter the mean of the log normal distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the standard deviation of the log normal distribution
    std_log_normal = st.sidebar.number_input(
        label='Enter the standard deviation of the log normal distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create log normal distribution datasets with user inputs
    dist_log_normal = np.random.lognormal(
        mean_log_normal, std_log_normal, num_samples)

    # create a plot of the log normal distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the log normal distribution
        fig = px.bar(x=dist_log_normal, title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the log normal distribution
        fig = px.line(y=dist_log_normal, title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the log normal distribution
        fig = px.scatter(y=dist_log_normal, title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the log normal distribution
        fig = px.histogram(x=dist_log_normal, title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the log normal distribution
        fig = px.box(x=dist_log_normal, points='all',
                     title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the log normal distribution
        fig = px.violin(x=dist_log_normal, box=True,
                        points='all', title='Log Normal Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the log normal distribution
        fig = px.density_contour(
            x=dist_log_normal, title='Log Normal Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Uniform':
    # create number input for the lower bound of the uniform distribution
    lower_bound_uniform = st.sidebar.number_input(
        label='Enter the lower bound of the uniform distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the upper bound of the uniform distribution
    upper_bound_uniform = st.sidebar.number_input(
        label='Enter the upper bound of the uniform distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create uniform distribution datasets with user inputs
    dist_uniform = np.random.uniform(
        lower_bound_uniform, upper_bound_uniform, num_samples)

    # create a plot of the uniform distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the uniform distribution
        fig = px.bar(x=dist_uniform, title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the uniform distribution
        fig = px.line(y=dist_uniform, title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the uniform distribution
        fig = px.scatter(y=dist_uniform, title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the uniform distribution
        fig = px.histogram(x=dist_uniform, title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the uniform distribution
        fig = px.box(x=dist_uniform, points='all',
                     title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the uniform distribution
        fig = px.violin(x=dist_uniform, box=True, points='all',
                        title='Uniform Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the uniform distribution
        fig = px.density_contour(y=dist_uniform, title='Uniform Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Triangular':
    # create number input for the lower bound of the triangular distribution
    st.warning(
        'you will need to enter a value in range (lower bound <= mode <= upper bound)')
    lower_bound_triangular = st.sidebar.number_input(
        label='Enter the lower bound of the triangular distribution', min_value=0.0, max_value=1.0, value=.4)
    # create number input for the mode of the triangular distribution
    mode_triangular = st.sidebar.number_input(
        label='Enter the mode of the triangular distribution', min_value=0.0, max_value=1.0, value=.45)
    # create number input for the upper bound of the triangular distribution
    upper_bound_triangular = st.sidebar.number_input(
        label='Enter the upper bound of the triangular distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create triangular distribution datasets with user inputs
    dist_triangular = np.random.triangular(
        lower_bound_triangular, mode_triangular, upper_bound_triangular, num_samples)

    # create a plot of the triangular distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the triangular distribution
        fig = px.bar(x=dist_triangular, title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the triangular distribution
        fig = px.line(y=dist_triangular, title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the triangular distribution
        fig = px.scatter(y=dist_triangular, title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the triangular distribution
        fig = px.histogram(x=dist_triangular, title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the triangular distribution
        fig = px.box(x=dist_triangular, points='all',
                     title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the triangular distribution
        fig = px.violin(x=dist_triangular, box=True,
                        points='all', title='Triangular Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the triangular distribution
        fig = px.density_contour(
            y=dist_triangular, title='Triangular Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Beta':
    # create number input for the alpha parameter of the beta distribution
    alpha_beta = st.sidebar.number_input(
        label='Enter the alpha parameter of the beta distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the beta parameter of the beta distribution
    beta_beta = st.sidebar.number_input(
        label='Enter the beta parameter of the beta distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create beta distribution datasets with user inputs
    dist_beta = np.random.beta(alpha_beta, beta_beta, num_samples)

    # create a plot of the beta distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the beta distribution
        fig = px.bar(x=dist_beta, title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the beta distribution
        fig = px.line(y=dist_beta, title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the beta distribution
        fig = px.scatter(y=dist_beta, title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the beta distribution
        fig = px.histogram(x=dist_beta, title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the beta distribution
        fig = px.box(x=dist_beta, points='all', title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the beta distribution
        fig = px.violin(x=dist_beta, box=True, points='all',
                        title='Beta Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the beta distribution
        fig = px.density_contour(y=dist_beta, title='Beta Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Gamma':
    # create number input for the shape parameter of the gamma distribution
    shape_gamma = st.sidebar.number_input(
        label='Enter the shape parameter of the gamma distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the scale parameter of the gamma distribution
    scale_gamma = st.sidebar.number_input(
        label='Enter the scale parameter of the gamma distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create gamma distribution datasets with user inputs
    dist_gamma = np.random.gamma(shape_gamma, scale_gamma, num_samples)

    # create a plot of the gamma distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the gamma distribution
        fig = px.bar(x=dist_gamma, title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the gamma distribution
        fig = px.line(y=dist_gamma, title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the gamma distribution
        fig = px.scatter(y=dist_gamma, title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the gamma distribution
        fig = px.histogram(x=dist_gamma, title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the gamma distribution

        fig = px.box(x=dist_gamma, points='all', title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the gamma distribution
        fig = px.violin(x=dist_gamma, box=True, points='all',
                        title='Gamma Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the gamma distribution
        fig = px.density_contour(y=dist_gamma, title='Gamma Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Chi-Squared':
    # create number input for the degrees of freedom of the chi-squared distribution
    df_chi = st.sidebar.number_input(
        label='Enter the degrees of freedom of the chi-squared distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create chi-squared distribution datasets with user inputs
    dist_chi = np.random.chisquare(df_chi, num_samples)

    # create a plot of the chi-squared distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the chi-squared distribution
        fig = px.bar(x=dist_chi, title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the chi-squared distribution
        fig = px.line(y=dist_chi, title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the chi-squared distribution
        fig = px.scatter(y=dist_chi, title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of
        fig = px.histogram(x=dist_chi, title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the chi-squared distribution
        fig = px.box(x=dist_chi, points='all',
                     title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the chi-squared distribution
        fig = px.violin(x=dist_chi, box=True, points='all',
                        title='Chi-Squared Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the chi-squared distribution
        fig = px.density_contour(y=dist_chi, title='Chi-Squared Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'F-Distribution':
    # create number input for the degrees of freedom of the F-distribution
    df_f = st.sidebar.number_input(
        label='Enter the degrees of freedom of the F-distribution', min_value=1.0, max_value=10.0, value=1.0)
    # create number input for the degrees of freedom of the F-distribution
    df_f2 = st.sidebar.number_input(
        label='Enter the degrees of freedom of the F-distribution', min_value=1.0, max_value=10.0, value=2.0)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create F-distribution datasets with user inputs
    dist_f = np.random.f(df_f, df_f2, num_samples)
    st.write(dist_f[:10])
    # create a plot of the F-distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the F-distribution
        fig = px.bar(y=dist_f, title='F-Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the F-distribution
        fig = px.line(y=dist_f, title='F-Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the F-distribution
        fig = px.scatter(y=dist_f, title='F-Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the F-distribution
        fig = px.histogram(y=dist_f, title='F-Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the F-distribution
        fig = px.box(x=dist_f, points='all',
                     title='F-Distribution')
    elif graph_type == 'Violin Plot':
        # create a violin plot of the F-distribution
        # create violin plot with customuzed attributes
        fig = px.violin(x=dist_f, box=True, points='all',
                        title='F-Distribution')
        st.plotly_chart(fig)

    elif graph_type == 'Density Plot':
        # create a density plot of the F-distribution
        fig = px.density_contour(y=dist_f, title='F-Distribution')
        st.plotly_chart(fig)

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
    dist_binomial = np.random.binomial(num_trials, prob_success, num_samples)

    # create a plot of the binomial distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the binomial distribution
        fig = px.bar(x=dist_binomial, title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the binomial distribution
        fig = px.line(y=dist_binomial, title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the binomial distribution
        fig = px.scatter(y=dist_binomial, title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the binomial distribution
        fig = px.histogram(x=dist_binomial, title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the binomial distribution
        fig = px.box(x=dist_binomial, points='all',
                     title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the binomial distribution
        fig = px.violin(x=dist_binomial, box=True, points='all',
                        title='Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the binomial distribution
        fig = px.density_contour(
            y=dist_binomial, title='Binomial Distribution')
        st.plotly_chart(fig)

elif distribution_type == 'Exponential':
    # create number input for the rate parameter of the exponential distribution
    rate_exp = st.sidebar.number_input(
        label='Enter the rate parameter of the exponential distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create exponential distribution datasets with user inputs
    dist_exp = np.random.exponential(rate_exp, num_samples)

    # create a plot of the exponential distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the exponential distribution
        fig = px.bar(x=dist_exp, title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the exponential distribution
        fig = px.line(y=dist_exp, title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the exponential distribution
        fig = px.scatter(y=dist_exp, title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the exponential distribution
        fig = px.histogram(x=dist_exp, title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the exponential distribution
        fig = px.box(x=dist_exp, points='all',
                     title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the exponential distribution
        fig = px.violin(x=dist_exp, box=True, points='all',
                        title='Exponential Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the exponential distribution
        fig = px.density_contour(y=dist_exp, title='Exponential Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Poisson':
    # create number input for the rate parameter of the poisson distribution
    rate_pois = st.sidebar.number_input(
        label='Enter the rate parameter of the poisson distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create poisson distribution datasets with user inputs
    dist_pois = np.random.poisson(rate_pois, num_samples)

    # create a plot of the poisson distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the poisson distribution
        fig = px.bar(x=dist_pois, title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the poisson distribution
        fig = px.line(y=dist_pois, title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the poisson distribution
        fig = px.scatter(y=dist_pois, title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the poisson distribution
        fig = px.histogram(x=dist_pois, title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the poisson distribution
        fig = px.box(x=dist_pois, points='all', title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the poisson distribution
        fig = px.violin(x=dist_pois, box=True, points='all',
                        title='Poisson Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the poisson distribution
        fig = px.density_contour(y=dist_pois, title='Poisson Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Bernoulli':
    # create number input for the probability of success of the bernoulli distribution
    prob_bern = st.sidebar.number_input(
        label='Enter the probability of success of the bernoulli distribution', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create bernoulli distribution datasets with user inputs
    # bernoilli distribution is a discrete distribution. it is different from the binomial distribution because it is not a continuous distribution

    dist_bern = stats.bernoulli(prob_bern).rvs(num_samples)

    # create a plot of the bernoulli distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the bernoulli distribution
        fig = px.bar(x=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the bernoulli distribution
        fig = px.line(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the bernoulli distribution
        fig = px.scatter(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the bernoulli distribution
        fig = px.histogram(x=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the bernoulli distribution
        fig = px.box(x=dist_bern, points='all', title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the bernoulli distribution
        fig = px.violin(x=dist_bern, box=True, points='all',
                        title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the bernoulli distribution
        fig = px.density_contour(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)

    # create a plot of the bernoulli distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the bernoulli distribution
        fig = px.bar(x=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the bernoulli distribution
        fig = px.line(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the bernoulli distribution
        fig = px.scatter(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the bernoulli distribution
        fig = px.histogram(x=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the bernoulli distribution
        fig = px.box(x=dist_bern, points='all', title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the beroulli distribution
        fig = px.violin(x=dist_bern, box=True, points='all',
                        title='Bernoulli Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the bernoulli distribution
        fig = px.density_contour(y=dist_bern, title='Bernoulli Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Negative Binomial':
    # create number input for the number of successes
    num_success = st.sidebar.number_input(
        label='Enter the number of successes', min_value=0, max_value=1000000, value=1000)
    # create number input for the probability of success
    prob_success = st.sidebar.number_input(
        label='Enter the probability of success', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create negative binomial distribution datasets with user inputs
    dist_negbin = np.random.negative_binomial(
        num_success, prob_success, num_samples)

    # create a plot of the negative binomial distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the negative binomial distribution
        fig = px.bar(x=dist_negbin, title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the negative binomial distribution
        fig = px.line(y=dist_negbin, title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the negative binomial distribution
        fig = px.scatter(y=dist_negbin, title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the negative binomial distribution
        fig = px.histogram(
            x=dist_negbin, title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the negative binomial distribution
        fig = px.box(x=dist_negbin, points='all',
                     title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the negative binomial distribution
        fig = px.violin(x=dist_negbin, box=True, points='all',
                        title='Negative Binomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the negative binomial distribution
        fig = px.density_contour(
            y=dist_negbin, title='Negative Binomial Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Geometric':
    # create number input for the probability of success
    prob_geo = st.sidebar.number_input(
        label='Enter the probability of success', min_value=0.0, max_value=1.0, value=.5)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create geometric distribution datasets with user inputs
    dist_geo = np.random.geometric(prob_geo, num_samples)

    # create a plot of the geometric distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the geometric distribution
        fig = px.bar(x=dist_geo, title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the geometric distribution
        fig = px.line(y=dist_geo, title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the geometric distribution
        fig = px.scatter(y=dist_geo, title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the geometric distribution
        fig = px.histogram(x=dist_geo, title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the geometric distribution
        fig = px.box(x=dist_geo, points='all', title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the geometric distribution
        fig = px.violin(x=dist_geo, box=True, points='all',
                        title='Geometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the geometric distribution
        fig = px.density_contour(y=dist_geo, title='Geometric Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Hypergeometric':
    # create number input for the number of successes
    num_success = st.sidebar.number_input(
        label='Enter the number of successes', min_value=0, max_value=1000000, value=1000)
    # create number input for the number of failures
    num_failure = st.sidebar.number_input(
        label='Enter the number of failures', min_value=0, max_value=1000000, value=1000)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create hypergeometric distribution datasets with user inputs
    dist_hyper = np.random.hypergeometric(
        num_success, num_failure, num_samples)

    # create a plot of the hypergeometric distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the hypergeometric distribution
        fig = px.bar(x=dist_hyper, title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the hypergeometric distribution
        fig = px.line(y=dist_hyper, title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the hypergeometric distribution
        fig = px.scatter(y=dist_hyper, title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the hypergeometric distribution
        fig = px.histogram(x=dist_hyper, title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the hypergeometric distribution
        fig = px.box(x=dist_hyper, points='all',
                     title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the hypergeometric distribution
        fig = px.violin(x=dist_hyper, box=True, points='all',
                        title='Hypergeometric Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the hypergeometric distribution
        fig = px.density_contour(
            y=dist_hyper, title='Hypergeometric Distribution')
        st.plotly_chart(fig)
elif distribution_type == 'Multinomial':
    # create number input for the number of trials
    num_trials = st.sidebar.number_input(
        label='Enter the number of trials', min_value=0, max_value=1000000, value=1000)
    # create number input for the number of samples
    num_samples = st.sidebar.number_input(
        label='Enter the number of samples', min_value=0, max_value=1000000, value=1000)

    # create multinomial distribution datasets with user inputs
    dist_mult = np.random.multinomial(num_trials, num_samples)

    # create a plot of the multinomial distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the multinomial distribution
        fig = px.bar(x=dist_mult, title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the multinomial distribution
        fig = px.line(y=dist_mult, title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the multinomial distribution
        fig = px.scatter(y=dist_mult, title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the multinomial distribution
        fig = px.histogram(x=dist_mult, title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the multinomial distribution
        fig = px.box(x=dist_mult, points='all',
                     title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the multinomial distribution
        fig = px.violin(x=dist_mult, box=True, points='all',
                        title='Multinomial Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the multinomial distribution
        fig = px.density_contour(y=dist_mult, title='Multinomial Distribution')
        st.plotly_chart(fig)

elif distribution_type == 'Weibull':
    # create number input for the shape parameter
    shape = st.sidebar.number_input(
        label='Enter the shape parameter', min_value=0, max_value=1000000, value=1000)
    # create number input for the scale parameter
    scale = st.sidebar.number_input(
        label='Enter the scale parameter', min_value=0, max_value=1000000, value=1000)

    # create weibull distribution datasets with user inputs
    dist_weibull = np.random.weibull(shape, scale)

    # create a plot of the weibull distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the weibull distribution
        fig = px.bar(x=dist_weibull, title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the weibull distribution
        fig = px.line(y=dist_weibull, title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the weibull distribution
        fig = px.scatter(y=dist_weibull, title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the weibull distribution
        fig = px.histogram(x=dist_weibull, title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the weibull distribution
        fig = px.box(x=dist_weibull, points='all',
                     title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the weibull distribution
        fig = px.violin(x=dist_weibull, box=True, points='all',
                        title='Weibull Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the weibull distribution
        fig = px.density_contour(y=dist_weibull, title='Weibull Distribution')
        st.plotly_chart(fig)


elif distribution_type == 'Student-T':
    # create number input for the degrees of freedom
    df = st.sidebar.number_input(
        label='Enter the degrees of freedom', min_value=0, max_value=1000000, value=10)
    # create number input for the scale parameter
    scale = st.sidebar.number_input(
        label='Enter the scale parameter', min_value=0, max_value=1000000, value=100)

    # create student-t distribution datasets with user inputs
    dist_student = np.random.student_t(df, scale)

    # create a plot of the student-t distribution
    if graph_type == 'Bar Chart':
        # create a bar chart of the student-t distribution
        fig = px.bar(x=dist_student, title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Line Chart':
        # create a line chart of the student-t distribution
        fig = px.line(y=dist_student, title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Scatter Plot':
        # create a scatter plot of the student-t distribution
        fig = px.scatter(y=dist_student, title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        # create a histogram of the student-t distribution
        fig = px.histogram(x=dist_student, title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Box Plot':
        # create a box plot of the student-t distribution
        fig = px.box(x=dist_student, points='all',
                     title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Violin Plot':
        # create a violin plot of the student-t distribution
        fig = px.violin(x=dist_student, box=True, points='all',
                        title='Student-T Distribution')
        st.plotly_chart(fig)
    elif graph_type == 'Density Plot':
        # create a density plot of the student-t distribution
        fig = px.density_contour(
            y=dist_student, title='Student-T Distribution')
        st.plotly_chart(fig)


else:
    st.error('Please select a distribution type')

st.markdown("""
        **Created by :** [Mustafa Bozkaya](https://github.com/mustafabozkaya)
         [Kaggle](https://www.kaggle.com/mustafabozka)
         [GitHub source](https://github.com/ds-ml-dl-ai-solutions/projects/blob/main/central_limit_theorem/)
        """)
