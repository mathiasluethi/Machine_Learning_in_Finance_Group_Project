#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:58:32 2018

@author: thepanda
"""

###############################################################################################################
#                                               TASK                                                          # 
###############################################################################################################


### MUST BE PART OF OUR PROJECT
# LOGISTIC REGRESSION
# RANDOM FOREST
# SUPPORT VECTOR MACHINE


# groups are to use a data set with financial ratios to forecast a company's future stock movement 
# 1, 3, 6 or 12 months down the road

# The data for this group project is to be taken from the WRDS data base (Wharton...)
# https://wrds-web.wharton.upenn.edu/wrds/mywrds/tou_acceptance.cfm
# malearn, Eiger2018

# Monthly data is to be considered 
# The observation period should be no longer than Jan-2006 to Dec-2015 (10 years) (can be shorter at discretion of group)
# last month considered has to be Dec-2015 !

# Use (at least) the 30 companies of the Dow Jones Industrial Index
# The respective list (incl. PERMNO  codes)  is  provided  on  OLAT
# It  is  up  to  the  group  to  use  data  of  other/more companies to better train the model.  
# However, the forecasting results relevant are only those of the 30 given Dow Jones companies I

# use a historic data set of financial ratios and stock prices 
# to train a model that is able to project a stock's future movement (up/down) most adequately

##### CENTRAL POINTS:
# Selection of relevant ratios: is the model better with many or less ratios? and which ratios?
# Forecast period: Take recent ratios or older ratios / ratios over a longer period?
# Feature engineering: Does adding of other features (besides financial ratios, like trading volume, bid/ask spread, previous ratios [autocorrelation] improve projection of prices?)

##### MODEL MUST: predict future stock movement: up/down => categorical variable

###############################################################################################################
#                                               PACKAGES / LIBRARIES                                          # 
###############################################################################################################

import numpy as np
import pandas as pd
import matplotlib as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn as skl
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network
from sklearn import neighbors
from functools import reduce
from sklearn.svm import SVR
from sklearn import neighbors
from functools import reduce
from sklearn.model_selection import train_test_split
import os
import datetime as dt

# Define static variables
___OTHERPC___ = 'C:/Users/Andrea Luca Lampart/Documents/Barbara/from Wharton_B_query1/wharton_ml3.csv'
___THISPC___ = '/home/andy/barbara/github_ml_uzh/Machine_Learning_in_Finance_Group_Project/Data/wharton_ml3.csv'
___OTHERRATIOS___ = 'C:/Users/Andrea Luca Lampart/Documents/Barbara/from OLAT/Ratios.csv'
___THISRATIOS___ = '/home/andy/barbara/github_ml_uzh/Machine_Learning_in_Finance_Group_Project/Data/Ratios.csv'
___OTHEROUTPUT___ = 'C:/Users/Andrea Luca Lampart/Documents/Barbara/generated/'
___THISOUTPUT___ = '/home/andy/barbara/github_ml_uzh/Machine_Learning_in_Finance_Group_Project/Data/generated/'
___FILLSTRAT___ = 'mean'

###############################################################################################################
#                                                    DATA IMPORT                                              # 
###############################################################################################################


##### IMPORT .csv file from wharton university (query saved as ml_2 in wharton account) 
# with all 30 Dow Jones firms listed in the PERMNO file on OLAT
# chosen variables (not all because file was to big => query error):

wharton = pd.read_csv(___THISPC___ , sep = ',')

# Rename  columns in wharton file
wharton.rename(columns = {'date': 'DATE_w', 'PERMNO': 'PERMNO_w'}, inplace = True)

# SPREAD seems to contain no data (nan), so I'm calculating spread via BID - ASK 
wharton['SPREAD'] = wharton['BID'] - wharton['ASK']

# REMOVE DATA THAT IS (PBLY) NOT NEEDED
wharton = wharton.drop(['RET', 'RETX', 'DLAMT', 'DLPDT', 'DCLRDT', 'DLRETX', 'DLRET', 'DLPRC', 'PAYDT', 'RCRDDT', 
                           'SHRFLG', 'DISTCD', 'DIVAMT', 'FACPR', 
                           'ACPERM', 'ACCOMP', 'SHRENDDT', 'FACSHR', 'ALTPRCDT'], axis = 1)

##### IMPORT FINANCIAL RATIOS (downloaded from OLAT)
ratios_all = pd.read_csv(___THISRATIOS___, sep = ',', converters = {'permno': str})

# Rename columns in ratios file
ratios_all.rename(columns = {'permno': 'PERMNO_r', 'public_date': 'DATE_r'}, inplace = True)

# REMOVE DATA THAT IS (PBLY) NOT NEEDED
ratios_all = ratios_all.drop(['adate', 'qdate'], axis = 1)

###############################################################################################################
#                                                    DATA PREPARATION                                         #
###############################################################################################################

### DATE
# set date as datetime64
wharton['DATE_w'] = pd.to_datetime(wharton['DATE_w']).dt.date.astype('datetime64[ns]')
ratios_all['DATE_r'] = pd.to_datetime(ratios_all['DATE_r']).dt.date.astype('datetime64[ns]')

### PERMNO
# set PERMNO column as integer (int64) for ratios_all, because its type 'O'
ratios_all['PERMNO_r'] = ratios_all['PERMNO_r'].convert_objects(convert_numeric = True)

### RENAME COLUMNS AND SET PERMNO AND DATE AS INDEX
wharton.rename(columns = {'PERMNO_w': 'PERMNO', 'DATE_w': 'DATE'}, inplace = True)
ratios_all.rename(columns = {'PERMNO_r': 'PERMNO', 'DATE_r': 'DATE'}, inplace = True)

### WHARTON CALCULATE RETURNS COLUMN
# Create a new dataframe
grouped_by_permno = pd.DataFrame()

# Now follows a typical split-change-merge pattern to calculate the values for each PERMNO
# For each permno in df, do:
for df_key in wharton.groupby('PERMNO').groups:
    permno_group = wharton.groupby('PERMNO').get_group(df_key)
    permno_group['PRC_RET'] = np.log(permno_group['PRC']/ permno_group['PRC'].shift(1))
    grouped_by_permno = pd.concat([grouped_by_permno, permno_group])
wharton = grouped_by_permno

# Delete unused variables
del df_key, permno_group, grouped_by_permno

### Set 0, 1 for PRC_RET
def set_mov(mydata):
    if mydata['PRC_RET'] > 0:
        return 'UP'
    elif mydata['PRC_RET'] < 0:
        return 'DOWN'
    elif mydata['PRC_RET'] == 0:
        return 'UP'

wharton = wharton.assign(MOVEMENT = wharton.apply(set_mov, axis = 1))
wharton['PRC_MOV'] = wharton['MOVEMENT'].factorize()[0]
wharton = wharton.drop('MOVEMENT', axis =1)

# Set all dates to the first day of the month
wharton['DATE'] = wharton['DATE'].apply(lambda dt: dt.replace(day=1))
ratios_all['DATE'] = ratios_all['DATE'].apply(lambda dt: dt.replace(day=1))


### JOIN DATASETS, OUTER
# Set index
wharton = wharton.set_index(['PERMNO', 'DATE'])
ratios_all = ratios_all.set_index(['PERMNO', 'DATE'])

# join datasets
joined_dataset = wharton.join(ratios_all, how = 'outer')


###################=====>
# delete datasets we do not need anymore
del wharton, ratios_all


# reset index
joined_dataset = joined_dataset.reset_index()

grouped_by_permno = pd.DataFrame()

# Group by PERMNO-Code and then remove first (0) row because it is nan in every PRC_RET group
for df_key in joined_dataset.groupby('PERMNO').groups:
    permno_group = joined_dataset.groupby('PERMNO').get_group(df_key)
    permno_group = permno_group[1:]
    grouped_by_permno = pd.concat([grouped_by_permno, permno_group])
joined_dataset = grouped_by_permno

# Delete unused variables
del df_key, permno_group, grouped_by_permno

## Detect and delete rows with nan
#nans = joined_dataset.isnull()
#del nans


# Remove percentages in row "divyield" and divide with 100 (so its decimal percentage) with string split
joined_dataset['divyield'] = joined_dataset['divyield'].str.rstrip('%').astype('float')/100

## remove outlier
#joined_dataset = joined_dataset.drop(index=3365, axis = 0)


# New Column with lagged movement of prices, because we want to have lagged prices as response vector later
# because we want to forecast prices of tomorrow, not of today with our model
joined_dataset['NEXT_DAY_PREDICTION'] = joined_dataset['PRC_MOV']
grouped_by_permno = pd.DataFrame()

for df_key in joined_dataset.groupby('PERMNO').groups:
    permno_group = joined_dataset.groupby('PERMNO').get_group(df_key)
    permno_group['NEXT_DAY_PREDICTION'] = permno_group['NEXT_DAY_PREDICTION'].shift(-1)
    permno_group = permno_group[pd.notnull(permno_group['NEXT_DAY_PREDICTION'])]
    grouped_by_permno = pd.concat([grouped_by_permno, permno_group])
joined_dataset = grouped_by_permno

# Delete unused variables
del df_key, permno_group, grouped_by_permno

# Make date numeric because RF can not use otherwise
joined_dataset['DATE'] = joined_dataset['DATE'].astype('int64')

# save file so we can use it later
joined_dataset.to_csv(___THISOUTPUT___ + 'joined_dataset_ml.csv')

### VERSION 1: Fill missing value with sklearn IMPUTER (fills with mean)
imp = Imputer(missing_values=np.nan, strategy = ___FILLSTRAT___ , axis=0)
imputed_dataset = pd.DataFrame(imp.fit_transform(joined_dataset))
imputed_dataset.columns = joined_dataset.columns
imputed_dataset.index = joined_dataset.index

imputed_dataset.to_csv(___THISOUTPUT___ + 'imputed_dataset_ml.csv')

### VERSION 2: Remove all rpws with Nan
dropnan_dataset = joined_dataset.dropna()

# Save 
dropnan_dataset.to_csv(___THISOUTPUT___ + 'dropnan_dataset_ml.csv')

### END