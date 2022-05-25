import streamlit as st
import pandas as pd
import os


st.title('SF Trees')
st.write('This app analyses trees in San Francisco using'
         ' a dataset kindly provided by SF DPW')

data_path = os.path.join(os.path.dirname("__file__"), "tree_app/trees.csv")
print("data path", data_path)
trees_df = pd.read_csv('trees.csv')
st.write(trees_df.head())
# write group by dbh and count the number of trees in each group
st.dataframe(trees_df.groupby("dbh").count())
df_dbh_grouped = pd.DataFrame(trees_df.groupby(['dbh']).
                              count()['tree_id'])  # create a new dataframe with the grouped data and the count of trees in each group
df_dbh_grouped.columns = ['tree_count']
st.line_chart(df_dbh_grouped)  # crea
st.bar_chart(df_dbh_grouped)
# create a line chart, bar chart, and area chart of the dataframe
st.area_chart(df_dbh_grouped)


st.title('SF Trees')
st.write('This app analyses trees in San Francisco using'
         ' a dataset kindly provided by SF DPW')
trees_df = pd.read_csv('trees.csv')
# drop rows with missing values in longitude and latitude columns
trees_df = trees_df.dropna(subset=['longitude', 'latitude'])
trees_df = trees_df.sample(n=1000)  # randomly sample 1000 rows
# create a map of the dataframe with the latitude and longitude columns
st.map(trees_df)
