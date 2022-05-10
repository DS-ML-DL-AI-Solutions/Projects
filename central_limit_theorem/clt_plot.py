import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create central limit theorem plot for the exponential distribution
# create a random exponential distribution with scale 1 and size 1000
dist_exp = np.random.exponential(scale=1, size=1000)
st.write("exponential distribution", dist_exp)
# plot the histogram of the exponential distribution
plt.hist(dist_exp, bins=50, density=True)
st.pyplot()  # display the figure
plt.plot(dist_exp)
st.pyplot()  # display the figure
dist_mean = []

for i in range(1000):
    # calculate the sample mean of the exponential distribution
    # choose 100 samples from the exponential distribution
    dist_mean.append(np.mean(np.random.choice(
        dist_exp, size=100, replace=True)))
# print the sample mean of the exponential distribution
st.write("sample mean of exp_dist", np.mean(dist_mean))
# plot the sample mean of the exponential distribution
# display the figure with kde=True and bins=50 to get a density plot instead of a histogram
sns.distplot(dist_mean, kde=True, bins=50)
st.pyplot()  # display the figure
