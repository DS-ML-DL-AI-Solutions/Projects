from dataclasses import replace
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Central Limit Theorem with binomal distribution")
st.header("Binomial Distribution Plot for 1000 samples")
prob_succes=st.number_input("Enter the probability of coin flip", max_value=1, min_value=0,value=0.5)
# create binomal distribution n is the number of trials, p is the probability of success and size is the number of samples
binom_dist = np.random.binomial(n=1, p=prob_succes, size=1000)
st.write("binomial distribution", binom_dist)

# calculate the population mean of the binomial distribution
pop_binom_mean = np.mean(binom_dist)
st.write("pop. mean of binom_dist", pop_binom_mean)

# create an empty list to store the sample mean of the binomial distribution
sample_binom_mean = []
for i in range(1000):
    # calculate the sample mean of the binomial distribution
    # choose 100 samples from the binomial distribution
    sample_binom_mean.append(
        np.mean(np.random.choice(binom_dist, size=100, replace=True)))
# print the sample mean of the binomial distribution
st.write("sample mean of binom_dist", np.mean(sample_binom_mean))

# plotting the sample mean of the binomial distribution
# create a figure and axes object for the histogram plot with a size of 15x12
fig, axe = plt.subplots(figsize=(15, 12))
# plot the sample mean of the binomial distribution
axe.hist(sample_binom_mean, bins=50, density=True)
# set the title of the figure
axe.set_title("sample mean of binom_dist dustribütion")
st.pyplot(fig)  # display the figure

fig2, axe2 = plt.subplots(figsize=(15, 12))
# plot the sample mean of the binomial distribution
axe2 = plt.scatter(x=range(len(sample_binom_mean)), y=sample_binom_mean)
# set the title of the figure
axe2 = plt.title("sample mean of binom_dist dustribütion")
st.pyplot(fig2)  # display the figure
