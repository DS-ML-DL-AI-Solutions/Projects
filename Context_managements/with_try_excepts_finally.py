#https://realpython.com/python-with-statement/#:~:text=The%20Python%20with%20statement%20creates,finally%20statement.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # for os.path.isfile, os.path.isdir, os.path.exists, os.path.join, os.path.split, os.path.splitext, os.path.basename, os.path.dirname and os.path.abspath functions and for os.path.dirname function to get the directory name of the file path passed to the function as an argument and for os.path.abspath function to get the absolute path of the file path passed to the function as an argument. For more information on these functions, please visit the following link: https://docs.python.org/3/library/os.html and https://docs.python.org/3/library/os.path.html
import sys # for sys.exit() to exit the program when the user clicks the "Run" button in the Streamlit app and the program is not running in the background

#https://docs.python.org/3/library/pathlib.html
import pathlib # for creating a directory in the current working directory and to check if a directory exists or not and to get the current working directory path and to get the parent directory path
import glob # for glob.glob() to search for files in a directory and return a list of matching filenames in the same order as the filenames would be returned by os.listdir()
import shutil as sh # for deleting files and folders in a directory and for copying files and folders in a directory to another directory
import time
import datetime as dt  # for date and time functions 

st.title("Context Managements with try, except, finally")
st.header("This is a simple program to create a directory in the current working directory and to check if a directory exists or not and to get the current working directory path and to get the parent directory path")
st.subheader("The program will create a directory in the current working directory and will check if a directory exists or not and will get the current working directory path and will get the parent directory path")

with st.spinner("Creating a directory in the current working directory..."):
    # create a directory in the current working directory
    pathlib.Path("./directory").mkdir(parents=True, exist_ok=True)
    # check if a directory exists or not
    if os.path.isdir("./directory"):
        st.success("The directory exists")
    else:
        st.error("The directory does not exist")
    # get the current working directory path
    cwd = os.getcwd()
    st.write("The current working directory is:", cwd)
    # get the parent directory path
    parent_dir = os.path.dirname(cwd)
    st.write("The parent directory is:", parent_dir)
    