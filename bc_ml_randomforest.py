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


imputed_dataset = pd.read_csv(___THISOUTPUT___ + 'imputed_dataset_ml.csv', sep = ',')
dropnan_dataset = pd.read_csv(___THISOUTPUT___ + 'dropnan_dataset_ml.csv', sep = ',')


#############=====> all

###############################################################################################################
#                                    RANDOM FOREST FEATURE SELECTION                                          #
#                                    with VERSION 1: IMPUTED DATASET                                          #
###############################################################################################################

# Extract labels of features
labels_of_features_1 = imputed_dataset.columns[:-1]
type(labels_of_features_1)

# X1 is the feature matrix
X1 = imputed_dataset.iloc[:, :-1]
# y1 is the response vector
y1 = imputed_dataset.iloc[:, -1]

# Do the train - test- split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0, stratify = y1)

# Check if there is the approximately same percentage of '1' i both training and test response vector
y1_train.sum() / y1_train.size
y1_test.sum() / y1_test.size

# Standardization with sklearn StandardScaler => NOT NECESSARY FR RANDOM FOREST, BUT NOT BAD EITHER SO LETS PUT IT HERE FOR STRUCTURE MEANS
standard_scaler_1 = preprocessing.StandardScaler().fit(X1_train)
X1_train = standard_scaler_1.transform(X1_train)
X1_test = standard_scaler_1.transform(X1_test)

### Save output to csv
#imputed_dataset.to_csv(___THISOUTPUT___ + 'imputed_dataset_ml.csv')

### Build Forest
my_forest_1 = RandomForestClassifier(random_state = 1)
my_forest_1.max_depth = 8
my_forest_1.fit(X1_train, y1_train)

# Check features for their importance for the prediction
features_importances_1 = my_forest_1.feature_importances_
# sort features in line with their importance for the prediction
indices_1 = np.argsort(features_importances_1)[::-1]
# Compute Standard deviation along axis

# print best 15 features
i = 0
n = 15
for i in range(n):
    print('{0:2d} {1:7s} {2:6.4f}'.format(i+1, labels_of_features_1[indices_1[i]], features_importances_1[indices_1[i]]))
del i,n

# Remove a lot of variables that were foud useless in the RF process
imputed_dataset_f = imputed_dataset[['vwretx', 'DATE', 'sprtrn', 'vwretd', 'ewretd', 'ewretx', 'divyield', 'BID', 'PEG_trailing', 
                                   'pe_op_basic', 'cash_lt', 'PEG_1yrforward', 'ASKHI', 'ptb', 'PEG_ltgforward',
                                   'NEXT_DAY_PREDICTION']]



#############=====> all

###############################################################################################################
#                                    RANDOM FOREST FEATURE SELECTION                                          #
#                                    with VERSION 2: DATASET WITH DROPPED NAN                                 #
###############################################################################################################

# Extract labels of features
labels_of_features_2 = dropnan_dataset.columns[:-1]
type(labels_of_features_2)

# X1 is the feature matrix
X2 = dropnan_dataset.iloc[:, :-1]
# y1 is the response vector
y2 = dropnan_dataset.iloc[:, -1]

# Do the train - test- split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0, stratify = y2)

# Check if there is the approximately same percentage of '1' i both training and test response vector
y2_train.sum() / y2_train.size
y2_test.sum() / y2_test.size

# Standardization with sklearn StandardScaler => NOT NECESSARY FR RANDOM FOREST, BUT NOT BAD EITHER SO LETS PUT IT HERE FOR STRUCTURE MEANS
standard_scaler_2 = preprocessing.StandardScaler().fit(X2_train)
X2_train = standard_scaler_2.transform(X2_train)
X2_test = standard_scaler_2.transform(X2_test)

### Save output to csv
#imputed_dataset.to_csv(___THISOUTPUT___ + 'imputed_dataset_ml.csv')

### Build Forest
my_forest_2 = RandomForestClassifier(random_state = 1)
my_forest_2.max_depth = 8
my_forest_2.fit(X2_train, y2_train)

# Check features for their importance for the prediction
features_importances_2 = my_forest_2.feature_importances_
# sort features in line with their importance for the prediction
indices_2 = np.argsort(features_importances_2)[::-1]
# Compute Standard deviation along axis

# print best 15 features
i = 0
n = 15
for i in range(n):
    print('{0:2d} {1:7s} {2:6.4f}'.format(i+1, labels_of_features_2[indices_2[i]], features_importances_2[indices_2[i]]))
del i,n

# Remove a lot of variables that were foud useless in the RF process
dropnan_dataset_f = dropnan_dataset[['vwretd', 'ptb', 'vwretx', 'DATE', 'CAPEI', 'ewretd', 'ewretx', 
                                   'PEG_1yrforward', 'accrual', 'pretret_earnat', 'ps', 'roe', 'pcf', 'pe_op_dil', 'BID',
                                   'NEXT_DAY_PREDICTION']]


#############=====> all

###############################################################################################################
#                                    RANDOM FOREST CLASSIFICATION                                             #
#                                    with VERSION 1: IMPUTED DATASET                                           #
###############################################################################################################

### Prepare Sets

# Extract labels of features
labels_of_features_1 = imputed_dataset_f.columns[:-1]
# X1 is the feature matrix
X1 = imputed_dataset_f.iloc[:, :-1]
# y1 is the response vector
y1 = imputed_dataset_f.iloc[:, -1]

# Do the train - test- split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0, stratify = y1)

# CHeck if there is the approximately same percentage of '1' i both training and test response vector
y1_train.sum() / y1_train.size
y1_test.sum() / y1_test.size



########## FOREST ##########

my_forest_1 = RandomForestClassifier(random_state = 1)
my_forest_1.max_depth = 8
my_forest_1.fit(X1_train, y1_train)

# Check features for their importance for the prediction
features_importances_1 = my_forest_1.feature_importances_
# sort features in line with their importance for the prediction
indices_1 = np.argsort(features_importances_1)[::-1]

# print best 15 features to see how it looks now
i = 0
n = 15
for i in range(n):
    print('{0:2d} {1:7s} {2:6.4f}'.format(i+1, labels_of_features_1[indices_1[i]], features_importances_1[indices_1[i]]))
del i,n

# Test prediction of y1 with the test feature matrix: gives the prediction vector
prediction_1 = my_forest_1.predict(X1_test)

# Calculate percentage of of ones in the test response vector
y_test_1_score = print('Ratio of Ones (Test 1) =  ' + str(y1_test.sum() / y1_test.size))
# Just to be sure the ones are distributed ca.the same in test and train response vector, check this:
y_train_1_score = print('Ratio of Ones (Training 1) =  ' + str(y1_train.sum() / y1_train.size))
# Calculate precentage of ones predicted with the model
prediction_1_score = print('Score (Prediction 1) =  ' + str(prediction_1.sum() / prediction_1.size))

# Calculate the score surplus above the test-set response vector score
(prediction_1.sum()/prediction_1.size) - (y1_test.sum()/y1_test.size)



###############################################################################################################
#                                    RANDOM FOREST CLASSIFICATION                                             #
#                                    with VERSION 2: DATASET WITH DROPPED NANs                                #
###############################################################################################################

### Prepare Sets

# Extract labels of features
labels_of_features_2 = dropnan_dataset_f.columns[:-1]
# X1 is the feature matrix
X2 = dropnan_dataset_f.iloc[:, :-1]
# y1 is the response vector
y2 = dropnan_dataset_f.iloc[:, -1]

# Do the train - test- split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0, stratify = y2)

# CHeck if there is the approximately same percentage of '1' i both training and test response vector
y2_train.sum() / y2_train.size
y2_test.sum() / y2_test.size



########## FOREST ##########

my_forest_2 = RandomForestClassifier(random_state = 1)
my_forest_2.max_depth = 8
my_forest_2.fit(X2_train, y2_train)

# Check features for their importance for the prediction
features_importances_2 = my_forest_2.feature_importances_
# sort features in line with their importance for the prediction
indices_2 = np.argsort(features_importances_2)[::-1]

# print best 15 features to see how it looks now
i = 0
n = 15
for i in range(n):
    print('{0:2d} {1:7s} {2:6.4f}'.format(i+1, labels_of_features_2[indices_2[i]], features_importances_2[indices_2[i]]))
del i,n

# Test prediction of y1 with the test feature matrix: gives the prediction vector
prediction_2 = my_forest_2.predict(X2_test)

# Calculate percentage of of ones in the test response vector
y_test_2_score = print('Ratio of Ones (Test 2) =  ' + str(y2_test.sum() / y2_test.size))
# Just to be sure the ones are distributed ca.the same in test and train response vector, check this:
y_train_2_score = print('Ratio of Ones (Train 2) =  ' + str(y2_train.sum() / y2_train.size))
# Calculate precentage of ones predicted with the model
prediction_2_score = print('Score (Prediction 2) =  ' + str(prediction_2.sum() / prediction_2.size))

# Calculate the score surplus above the test-set response vector score
(prediction_2.sum()/prediction_2.size) - (y2_test.sum()/y2_test.size)


