# EXAMPLE 11.11: THE GOOGLE-APP.PY PROGRAM
# Example 11.11
import yfinance as yf
import streamlit as st
from datetime import datetime
from datetime import date
import datetime as dt
import pandas as pd
import numpy as np
import altair as alt

st.write("""
#  Stock Price Streamlit App
Shown are the stock **closing price** and ***volume*** of a Company!
""")


@st.cache  # cache the dataframe for faster loading times next time the app is run
def load_data():  # load the data from wikipedia S&P500 and return it as a dataframe
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df


@st.cache  # st.cache is a decorator that caches the dataframe for faster loading times next time the app is run
def load_ftse100data():  # load the data from wikipedia FTSE100 and return it as a dataframe
    url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
    html = pd.read_html(url, header=0)
    df = html[3]
    return df


# set the start date to be one year ago from today
start = date.today()-dt.timedelta(days=365)
# sidebar title for the date range selection
st.sidebar.markdown("# Start and End Date:")
w1 = st.sidebar.date_input("Start", start, min_value=date(
    1970, 1, 1))  # sidebar date input for start date
st.write("Value 1:", w1)  # write the value of the start date to the app

w2 = st.sidebar.date_input("End", date.today(), min_value=date(
    1970, 1, 1))  # sidebar date input for end date
st.write("Value 1:", w2)  # write the value of the end date to the app
tickerSymbol = 'GOOGL'  # set the ticker symbol to be GOOGL
name = 'Google'  # set the name of the company to be Google
option = st.sidebar.selectbox(
    'Select a Stock Market:',
    ["S&P 500", "FTSE 100"], index=0)  # sidebar selection box for the stock market
# write the selected stock market to the app
st.sidebar.write('You selected:', option)

# if the user selects the S&P 500, load the S&P500 dataframe
if option == "S&P 500":

    df = load_data()

    print(df.describe())  # print the dataframe description
    st.write(df.head())  # write the dataframe head to the app
    # create stream table for the S&P500 dataframe
    st.table(df.head(20).style.format(
        {'Close': '${:.2f}', 'Volume': '{:,.0f}'}).hide_index())  # hide the index column and format the dataframe columns to 2 decimal places and comma seperated numbers for volume and close price
    # create a line chart for the S&P500 dataframe
    st.line_chart(df.CIK)  # create a line chart for the dataframe

    # coms = df.groupby('EPIC')
    # coms = df['Symbol'].unique()
    # names = df['Security'].unique()
    coms = df['Symbol']
    names = df['Security']
    # print(coms)

    # Sidebar - Sector selection
    # selected_scoms = st.sidebar.multiselect('Companies', coms , coms)

    option1 = st.sidebar.selectbox(
        'Select a Company:',
        coms, index=0)

    #st.sidebar.write('You selected:', option1)
    i = [i for i, j in enumerate(coms) if j == option1]
    #i = np.where(coms == option1)
    name = names[i]
    # print(i)
    # print(name)
    tickerSymbol = option1
    st.sidebar.write(option1)


elif option == "FTSE 100":

    df = load_ftse100data()
    # print(df)
    st.write(df.head())
    #coms = df.groupby('EPIC')
    coms = df['EPIC']
    names = df['Company']
    # print(coms)

    # Sidebar - Sector selection
    #selected_scoms = st.sidebar.multiselect('Companies', coms , coms )

    option2 = st.sidebar.selectbox(
        'Select a Company:',
        coms, index=0)
    #st.sidebar.write('You selected:', option2)
    i = [i for i, j in enumerate(coms) if j == option2]
    #i = np.where(coms == option2)
    name = names[i]
    tickerSymbol = option2
    st.sidebar.write(option2)

# https://towardsdatascience.com/how-to-get-stock-data-using-pythonc0de1df17e75

#define the ticker symbol
#tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol) 

tickerDf = tickerData.history(period='1d', start=w1, end=w2) # get the historical prices for this ticker for the selected date range
# Open High Low Close Volume Dividends Stock Splits
st.write(name)
st.write("""
## Closing Price
""")
if (len(tickerDf)>1):
 st.line_chart(tickerDf.Close)
 st.write("Percentage Change : " +str(int((tickerDf.Close[-1] -
tickerDf.Close[0])/tickerDf.Close[0]*1000)/10.0) + "%")
else:
 st.write("Data not available...")
st.write("""
## Volume Price
""")
if (len(tickerDf)>1):
 st.line_chart(tickerDf.Volume)
else:
 st.write("Data not available...")


toplist = []
if ( st.button("Top Performers")):
 #for i in coms:
 for i in range(len(coms)):


    tickerData = yf.Ticker(coms[i])
    #get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start=w1, end=w2)
    print(len(tickerDf))
    if (len(tickerDf)<1):
        continue
 percentChange = int((tickerDf.Close[-1] - tickerDf.Close[0])/
tickerDf.Close[0]*1000)/10.0
 name = names[i]
 toplist.append([coms[i],name,percentChange])
 print(coms[i])
 print(name)
 print(percentChange )
 
 results = sorted(toplist,key=lambda l:l[2], reverse=True) 
 st.write(results[0:19])
 #https://docs.streamlit.io/en/stable/api.html
 #chart_data = pd.DataFrame(
 # np.random.randn(50, 3),
 # columns=["a", "b", "c"])
 #st.bar_chart(chart_data)
 #https://discuss.streamlit.io/t/sort-the-bar-chart-in-descendingorder/1037/2
 #import streamlit as st
 #import pandas as pd
 #import altair as alt
 x = [x[0] for x in results[0:19]]
 print(x)
 y = [x[1] for x in results[0:19]]
 print(y)
 z = [x[2] for x in results[0:19]]
 print(z)
 data = pd.DataFrame({
 'Company Symbol': x,
 'Company Name': y,
 'Percentage Change': z,
 })
 st.write(data)
 st.write(alt.Chart(data).mark_bar().encode(
 x=alt.X('Company Symbol', sort=None),
 y='Percentage Change',
 ))
