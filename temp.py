# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Task description
# Monthly data is to be considered and the observation 
# period should be no longer than Jan-2006 to Dec-2015 
# (10 years). Going with less years is at the discretion 
# of the group. However, the last month considered has 
# to be Dec-2015.
# Use (at least) the 30 companies of the Dow Jones Industrial 
# Index. The respective list (incl. PERMNO codes1) is provided 
# on OLAT. It is up to the group to use data of other/more 
# companies to better train the model. However, the forecasting 
# results relevant are only those of the 30 given Dow Jones companies.


# Notes from class 
#How to proceed: Train/Test split --> scale this (1) 
# --> Train data --> Scale test based on (1)


# import packages & libraries
import pandas as pd
import numpy as np
import os
import csv
import matplotlib as pl
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# set working directory
os.chdir('/Users/pamelamatias/Dropbox/Introduction to machine learning/Data')

# display more rows
pd.set_option('display.max_rows', 80)

# read in ratios
data = pd.read_csv('Ratios.csv', sep=';')
df_ratios = pd.DataFrame(data)

# read in wharton file
data_2 = pd.read_csv('Wharton.csv', sep=';')
df_wharton = pd.DataFrame(data_2)

# delete empty Data
df_wharton = df_wharton.drop(['DLAMT', 'DLPDT', 'ACPERM', 'ACCOMP', 
                              'DLRETX', 'DLPRC', 'DLRET'], axis = 1)

# rename equal column headers in both files
df_ratios.rename(columns = {'permno': 'PERMNO_r', 'public_date': 'DATE_r'}, inplace = True)




##############################################################################

# From here only looking at the df_ratios file

# showing test scores (optional)
# df.describe()

# fill missing values/ forward-fill to propagate previous value forward
df_ratios = df_ratios.fillna(axis='rows', method='ffill')

# remove columns that are not necessary
df_ratios = df_ratios.drop(['adate', 'qdate'], axis=1)

# delete NaNs
df_ratios = df_ratios.dropna(axis=0, how='any')

# we want just floats, therefore:
df_ratios.select_dtypes(['object'])
df_ratios['divyield'] = df_ratios['divyield'].str.rstrip('%').astype('float')/100.0
   
# we do not want to consider the first two columns
df_ratios_small = df_ratios.loc[:,'CAPEI':]

# before applying PCA we need to scale the data
# first get all column names
col_names = list(df_ratios_small)

# Separating out the features
x = df_ratios_small.loc[:, col_names].values

# Standardizing the features (onto scale mean = 0 and variance = 1)
x = StandardScaler().fit_transform(x)

# which components to choose according PCA; we want 15 for example
pca = PCA(n_components=15)

principalComponents = pca.fit_transform(x)










