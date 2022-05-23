# penguin EDA applications

import io
import os
import time
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
        # st.header("Created by Mustafa Bozkaya")  # header of the app
        # subheader of the app
        st.subheader(
            "This is a simple program to explore the penguin species data")

        st.write("General  Data Information")  # write the data information
        # write the shape of the data
        st.success("The penguins data contains \n " + str(self.data_shape[0]) +
                   " rows and " + str(self.data_shape[1]) + " columns")  # write the shape of the data
        # write the data columns
        st.markdown("<h3>Data Columns is below</h3>", unsafe_allow_html=True)
        st.write(self.data_columns)  # write the data columns

        # create side bar menu for data visualization
        st.sidebar.title("Data Analysis Tools")  # sidebar title

        # sidebar options
        # sidebar_option = st.sidebar.selectbox(
        #     "Please select an option", ("Show data Top 10", "Show data types",
        #                                 "Show data summary", "Show data visualization", "Show data information"))  # select an option from the sidebar

        if st.sidebar.checkbox("Show data Top 10"):  # checkbox for data top 10
            st.dataframe(self.data.head(10))
        if st.sidebar.checkbox("Show data types"):  # checkbox for data types
            #print("dtata types info :", type(self.data_types))
            # self.data_types_df = pd.DataFrame(self.data_types)
            # st.dataframe(self.data_types_df.style.highlight_max(axis=0))
            data_types_df = self.numeric_exp(self.data)
            st.write(data_types_df)
        if st.sidebar.checkbox("Show data summary"):  # checkbox for data summary
            # write the data descriptive statistics (mean, std, min, max, 25%, 50%, 75%) for each column
            st.write("The penguins data descriptive statistics")
            st.write(self.data.describe())
            # self.data.info()  # write the data information (number of rows, number of columns, data types, data memory size, data memory usage, data memory usage percent, data memory usage percent relative to the total physical memory, data memory usage percent relative to the total swap memory, data memory usage percent relative to the total virtual memory)

        # checkbox for data visualization
        if st.sidebar.checkbox("Show data information"):
            buffer = io.StringIO()  # create a buffer object to store the dataframe information
            self.data.info(buf=buffer)  # write the data information (number of rows, number of columns, data types, data memory size, data memory usage, data memory usage percent, data memory usage percent relative to the total physical memory, data memory usage percent relative to the total swap memory, data memory usage percent relative to the total virtual memory) to the buffer object and then write the buffer object

            info_str = buffer.getvalue()  # get the dataframe information from the buffer object
            # split the dataframe information into a list
            # str = s.split('\n')

            info_df = pd.DataFrame([line.split()
                                   for line in info_str.split('\n')[3:-3]])

            # drop the column  index
            info_df.drop(info_df.columns[0], axis=1, inplace=True)

            info_df.columns = ['Columns', 'Not Null',
                               'Count', 'Dtype']  # rename the columns
            # drop index row
            info_df.drop(info_df.index[1], inplace=True)
            # reset the index of the dataframe and drop the old index
            info_df.reset_index(drop=True, inplace=True)
            # drop the first row of the dataframe (the dataframe information) and drop the old index of the dataframe

            st.write(info_df)
        # checkbox for data information
        if st.sidebar.checkbox("Show plot count of each feature in the data"):
            feature=st.selectbox("Select a feature",self.data.select_dtypes(exclude=['int','float']).columns)
            st.write("The penguins data visualization")
            # create a subplot
            fig, ax = plt.subplots(figsize=(10, 10))
            # plot the data
            sns.countplot(x=feature, data=self.data, ax=ax)
            # show the plot
            st.pyplot(fig)

        species_data = self.data["species"].unique().tolist()
        species_data.insert(0, "all")

        with st.spinner('Wait for it...'):
            time.sleep(2)

            st.success(f"penguıns species: {species_data}")

        check_viz = st.sidebar.checkbox("Show data visualization with Seaborn")
        if check_viz:
            # create selecbox for seaorn plot style
            select_style = st.sidebar.selectbox("Select a style", ["darkgrid", "whitegrid", "dark", "white", "rdbu", "ticks"],
                                                index=0)

            selected_species = st.sidebar.selectbox("Select a species to visualize",
                                                    species_data)

            # create a checkbox to select  islands to visualize
            select_island = st.sidebar.checkbox("Show penguins by island")

            select_sex = st.sidebar.checkbox(
                "Show penguins by gender")

            # write the selected species
            st.write("You selected:", selected_species)
            select_x = st.sidebar.selectbox("Select a column to plot on the x-axis",
                                            self.data.select_dtypes(include=['int64', 'float64']).columns)
            select_y = st.sidebar.selectbox("Select a column to plot on the y-axis",
                                            self.data.select_dtypes(include=['int64', 'float64']).columns)

            # create a scatter plot of the selected species and the selected columns on the x and y axis
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            # set the style of the plot
            sns.set(style=select_style)
            markers = {"Adelie": "X", "Chinstrap": "o", "Gentoo": "*"}
            sns.scatterplot(x=select_x, y=select_y,
                            data=self.data,
                            hue=("species" if selected_species !=
                                 "all" else None),
                            style=("island" if select_island else None),
                            markers=markers,
                            ax=axes)
            plt.legend(loc='upper right')
            plt.title("Penguins by Species")
            plt.xlabel(select_x)
            plt.ylabel(select_y)
            st.pyplot(fig)

            # create a bar plot of the selected species and the selected columns on the x and y axis
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            # set the style of the plot
            sns.set_style("whitegrid")
            sns.barplot(x=select_x, y=select_y,
                        data=self.data,
                        hue=("species" if selected_species !=
                             "all" else None),
                        ax=axes)
            st.pyplot(fig)

            if selected_species == "all":
                # create a bar chart to show the distribution of the species
                st.markdown("### The penguins species distribution")
                st.bar_chart(self.data["species"].value_counts())
                st.pyplot()
            else:
                st.bar_chart(
                    self.data[self.data["species"] == selected_species]["species"].value_counts())
                st.pyplot()
            # create markdown for copyright information and link to the source code repository of the app on GitHub

        st.markdown("""
        **Created by :** [Mustafa Bozkaya](https://www.linkedin.com/in/mustafa-bozkaya)
        **Data source:** [Penguins species data](https://www.kaggle.com/datasets/mustafabozka/palmers-penguins)
        **Code source:** [GitHub](https://github.com/ds-ml-dl-ai-solutions/projects/blob/main/penguin_app/penguin_EDA.py)
        """)

    def numeric_exp(self, data):

        df_types = pd.DataFrame(data.dtypes, columns=['Data_Type'])

        # select numeric columns
        numerical_cols = df_types[df_types['Data_Type'].isin(
            ['int64', 'float64'])].index.values

        st.success(f"numeric columns : {numerical_cols}")

        df_types['Count'] = data.count()
        df_types['Unique Values'] = data.nunique()

        df_types['Min'] = data[numerical_cols].min()

        df_types['Max'] = data[numerical_cols].max()

        df_types['Average'] = data[numerical_cols].mean()

        df_types['Median'] = data[numerical_cols].median()

        df_types['St. Dev.'] = data[numerical_cols].std()

        return df_types.astype(str)

    def infoOut(data, details=False):
        dfInfo = data.columns.to_frame(name='Column')
        dfInfo['Non-Null Count'] = data.notna().sum()
        dfInfo['Dtype'] = data.dtypes
        dfInfo.reset_index(drop=True, inplace=True)
        if details:
            rangeIndex = (dfInfo['Non-Null Count'].min(),
                          dfInfo['Non-Null Count'].min())
            totalColumns = dfInfo['Column'].count()
            dtypesCount = dfInfo['Dtype'].value_counts()
            totalMemory = dfInfo.memory_usage().sum()
            return dfInfo, rangeIndex, totalColumns, dtypesCount, totalMemory
        else:
            return dfInfo

# create load file function
# cache the function to avoid re-run the function every time the page is refreshed,
# arg1: the function to be cached and arg2: the name of the cache file to be created and stored in the cache folder
# allow_output_mutation=True means the function can change the output of the function without affecting the function itself ,
#  show_spinner=False means the function will not show the spinner when the function is running


@st.cache(allow_output_mutation=True)
def load_file(datafile):
    # wait for it...
    with st.spinner('Wait for file uploading...'):
        time.sleep(2)
    if datafile is not None:
        st.success("Data loaded successfully")
        df = pd.read_csv(datafile)
    else:
        data_path = os.path.join(os.path.dirname(__file__), "penguins.csv")
        df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    # load the data
    try:
        # upload the data file from the computer and read it as a pandas dataframe,it returns a pandas dataframe
        datafile = st.file_uploader("Upload your data file", type=["csv"])

        # load the data file and return a pandas dataframe
        df = load_file(datafile)
        # create an instance of the class
        app = Penguin_EDA(df)
    except Exception as e:
        st.error(e)
    # create the app
