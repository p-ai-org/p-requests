#!/usr/bin/env python
# coding: utf-8

'''
Code for piping the majority and minority classes to their appropriate models, and running the models
'''
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#Other model will go here as well
import LACER as lc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sgcrf import SparseGaussianCRF


#Slightly editied lacer funcions
def preprocessing(df, start_date, end_date):
    """
    Filters dataframe by specified start and end_dates and runs CleanedFrame on it.  
    """ 

    #Filter dataframe by dates 
    df = df[(df['Just Date'] >= start_date) & (df['Just Date'] <= end_date)]
    df = lc.CleanedFrame(df)

    return df
def lacer(df, df1, train_start_date, train_end_date, test_start_date, test_end_date, request_type, CD, predictor_num): #Once model is ready, replace df with csv
    """
    Trains 3 GCRF models on data from specified CD, Request Type, and Owner which is assigned to fulfill request. 
    Uses specified start and end dates for training and testing to creat train and test sets. 
    """

    #Create Training and Testing Sets
    dftrain = preprocessing(df , train_start_date, train_end_date)
    dftrain = dftrain.reset_index(drop = True)
    dftest = preprocessing(df1, test_start_date, test_end_date)
    dftest = dftest.reset_index(drop = True)

    #Reserve test set for training on all 3 models. 
    y_train, y_test = lc.CreateTestSet(dftest, predictor_num)
    y_test = y_test.reshape((-1, 1))


## 2 Models
    #Model1: CD
    modelCD = SparseGaussianCRF(lamL=0.1, lamT=0.1, n_iter=10000)
    dftrainCD = dftrain[dftrain['CD'] == CD].reset_index(drop = True)

    X_trainCD, X_testCD = lc.CreateTrainSet(dftrainCD, predictor_num)
    X_testCD = X_testCD.reshape((-1, 1))
    modelCD.fit(X_trainCD, X_testCD)

    y_predCD = modelCD.predict(y_train)

    #Model2: Request_type
    modelRT = SparseGaussianCRF(lamL=0.1, lamT=0.1, n_iter=10000)
    dftrainRT = dftrain[dftrain['RequestType'] == request_type].reset_index(drop = True)

    X_trainRT, X_testRT = lc.CreateTrainSet(dftrainRT, predictor_num)
    X_testRT = X_testRT.reshape((-1, 1))

    modelRT.fit(X_trainRT, X_testRT)

    y_predRT = modelRT.predict(y_train)


    #Average out all predictions
    y_predFinal = (y_predCD + y_predRT )/2

    #Return metrics 
    return lc.metrics(y_predFinal, y_test)

"""
Records whether or not a number is greater than 7. 
"""
def gelev(val): 
    if val <= 11: 
        return 0
    else: 
        return 1

'''
Preprocessing function. Takes in the file path to the data and loads it in a DataFrame, then calcuates the elapsed days per request and marks them as more than or less than eleven days. Then it encodes the appropriate values and returns the train data, labels, and the formatted dataframe.
'''
def preprocess(df_path):
    df = pd.read_csv(df_path)
    df['Just Date'] = df['Just Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df['Eleven'] = df['ElapsedDays'].apply(gelev, 0)
    #Encode values
    c = ['Anonymous','AssignTo', 'RequestType', 'RequestSource','CD','Direction', 'ActionTaken', 'APC' ,'AddressVerified']
    d = ['Latitude', 'Longitude']
    #Put desired columns into dataframe, drop nulls. 
    dfn = df.filter(items = c + d + ['ElapsedDays'])
    dfn = dfn.dropna()
    #Separate data into explanatory and response variables
    XCAT = dfn.filter(items = c).values
    XNUM = dfn.filter(items = d).values
    y = dfn['ElapsedDays'] <= 11
    #Encode cateogrical data and merge with numerical data
    labelencoder_X = LabelEncoder()
    for num in range(len(c)): 
        XCAT[:, num] = labelencoder_X.fit_transform(XCAT[:, num])
    onehotencoder = OneHotEncoder()
    XCAT = onehotencoder.fit_transform(XCAT).toarray()
    X = np.concatenate((XCAT, XNUM), axis=1)
    return X,y, dfn

'''
Runs the model that classifies each request as more than or less than/equal to 11 days. Parameters are the hyperparameters for the model itself, and the train data and labels.
'''
def estimation_model(estimators, depth,X,y):    
    rf = RandomForestClassifier(n_estimators = estimators, max_depth = depth)
    print('creating train, test, val split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
    #Train Model
    print('training model')
    rf.fit(X_train, y_train)
    #Test model
    print('testing model')
    y_vpred = rf.predict(X_val)
    #Print Accuracy Function results
    print("Accuracy:",metrics.accuracy_score(y_val, y_vpred))
    print("Precision, Recall, F1Score:",metrics.precision_recall_fscore_support(y_val, y_vpred, average = 'binary'))
    return rf

'''
Will be removed
'''
def dummy_model(df):
    return 0,0

'''
Takes a file path to the data, runs the appropriate preprocessing steps, and uses the model to classify everything into the majority and minority class. Returns a dataframe with the majority class and a separate one with the minority class.
'''
def split_to_models(df_file_path):
    print('Calculating train data and labels')
    X, y, df = preprocess(df_file_path)
    df_clean = pd.read_csv(df_file_path)
    print('Creating 11 day classifier')
    model_eleven = estimation_model(50,20,X,y)
    df['LessEqualEleven'] = model_eleven.predict(X)
    df['LessEqualEleven'] = df['LessEqualEleven'].apply(lambda x: int(x))
    df_sgcrf = df[df['LessEqualEleven'] == 1.0]
    df_other = df[df['LessEqualEleven'] == 0.0]
    return df_sgcrf.merge(df_clean,on=['Latitude','Longitude']), df_other
'''
Takes the data file path and parameters for the lacer model and runs the pipeline to get the appropriate dataframes, then runs both models. This is the main function you should run from this file.
'''
def run_split_models(df_file_path,train_start_date, train_end_date, test_start_date, test_end_date, request_type, CD, predictor_num):
    df_sgcrf, df_other = split_to_models(df_file_path)
    print('running SGCRF')
    rmse, mae = lacer(df_sgcrf.copy(),df_sgcrf.copy(), train_start_date, train_end_date, test_start_date, test_end_date, request_type, CD, predictor_num)
    print('running (other model)')
    a,b = dummy_model(df_other)
    return (rmse,mae),(a,b)


