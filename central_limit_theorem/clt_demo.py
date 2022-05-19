import streamlit as st

import numpy as np

import matplotlib.pyplot as plt


st.title('Illustrating the Central Limit Theorem with Streamlit')
st.subheader('Created by Mustafa Bozkaya')
st.write(('This app simulates a thousand coin flips using the chance of heads input below,'
          'and then samples with replacement from that population and plots the histogram of the'
          ' means of the samples, in order to illustrate the Central Limit Theorem!'))
st.markdown('<h1>This is a simple program to create a plot of the **Central Limit Theorem** with the **binomal distribution**</h1>', unsafe_allow_html=True)

# create number input for the probability of success
perc_heads = st.number_input(
    label='Chance of Coins Landing on Heads', min_value=0.0, max_value=1.0, value=.5)
# create text input for graph title
graph_title = st.text_input(label='Title of the Graph', value='Coin Flip Distribution', type='default',
                            placeholder='Enter the title of the graph', max_chars=100)
#create data frame for the binomial distribution
binom_dist = np.random.binomial(n=1, p=perc_heads, size=1000) # n is the number of trials, p is the probability of success and size is the number of samples


list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(
        binom_dist, 100, replace=True).mean())


fig, ax = plt.subplots()
ax = plt.hist(list_of_means)

st.pyplot(fig)
