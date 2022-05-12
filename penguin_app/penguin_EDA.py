# penguin EDA applications

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


class Penguin_EDA:
    def __init__(self, data):  # data is a pandas dataframe object
        self.data = data
        self.data_shape = self.data.shape
        self.data_columns = self.data.columns
        self.data_types = self.data.dtypes
        self.initapp()  # initialize the app

    def initapp(self):
        # title of the app
        st.title("Penguins species Exploration Data Analysis")
        st.header("Created by Mustafa Bozkaya")  # header of the app
        # subheader of the app
        st.subheader(
            "This is a simple program to explore the penguin species data")

        st.write("penguins data Information")  # write the data information
        # write the shape of the data
        st.success("The penguins data contains \n " + str(self.data_shape[0]) +
                   " rows and " + str(self.data_shape[1]) + " columns")  # write the shape of the data
        # write the data columns
        st.write("The penguins data contains the following columns:")
        st.write(self.data_columns)  # write the data columns
        # write the data information
        st.write("The penguins data descriptive statistics")

        # write the data descriptive statistics (mean, std, min, max, 25%, 50%, 75%) for each column
        st.write(self.data.describe())
        # self.data.info()  # write the data information (number of rows, number of columns, data types, data memory size, data memory usage, data memory usage percent, data memory usage percent relative to the total physical memory, data memory usage percent relative to the total swap memory, data memory usage percent relative to the total virtual memory)

        buffer = io.StringIO()  # create a buffer object to store the dataframe information
        self.data.info(buf=buffer)
        s = buffer.getvalue()  # get the dataframe information from the buffer object
        # split the dataframe information into a list
        s = s.split('\n')
        st.write("The penguins data information")  # write the data information
        st.write(s)  # write the data information


if __name__ == "__main__":
    # load the data
    data = pd.read_csv("penguins.csv")  # load the data from the csv file
    # create the app
    app = Penguin_EDA(data)  # create the app
