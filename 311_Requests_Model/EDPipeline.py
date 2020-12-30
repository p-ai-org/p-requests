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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import LACER as lc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sgcrf import SparseGaussianCRF
import pickle
from joblib import dump, load
import lagcrf
import psycopg2

#Features to use in our model, c = categorical, d = numeric
c = ['AssignTo', 'RequestType', 'RequestSource', 'Month', 'Anonymous', 'CreatedByUserOrganization']
d = ['Latitude', 'Longitude']
    
"""
Return whether or not a number is greater than 11. 
"""
def gelev(val): 
    if val <= 11: 
        return 0
    else: 
        return 1

'''
Preprocessing function. Takes in the file path to the data and loads it in a DataFrame, 
then calcuates the elapsed days per request and marks them as more than or less than eleven days. 
Then it encodes the appropriate values and returns the train data, labels, and the formatted dataframe.
If formatted is False, it will convert the Just Date column into datetime objects
If encode is false, it requires that the onehotencoder has already been dumped into a joblib file,
so make sure that this has been run once on all the data with encode equal to true.
'''
def preprocess(df, formatted=False,encode=False):
    if not formatted:
        df['Just Date'] = df['Just Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df['Eleven'] = df['ElapsedDays'].apply(gelev, 0)
    df['Month'] = df['Just Date'].dt.month_name()
    #Put desired columns into dataframe, drop nulls. 
    dfn = df.filter(items = c + d + ['ElapsedDays','Just Date','CreatedDate','ClosedDate','CD'])
    dfn = dfn.dropna()
    #Separate data into explanatory and response variables
    XCAT = dfn.filter(items = c).values
    XNUM = dfn.filter(items = d).values
    y = dfn['ElapsedDays'] <= 11
    #Encode cateogrical data and merge with numerical data
    if encode:
        onehotencoder = OneHotEncoder()
        print(XCAT.shape)
        onehotencoder.fit_transform(XCAT)
        X = np.concatenate((XCAT, XNUM), axis=1)
        print()
        dump(onehotencoder,'onehot.joblib')
    else:
        onehotencoder = load('onehot.joblib')
        XCAT = onehotencoder.transform(XCAT).toarray()
        X = np.concatenate((XCAT, XNUM), axis=1)
    return X,y, dfn

'''
Preprocess a request given as a dataframe
'''
def preprocess_request(df):
    dfn = df.filter(items = c + d)
    dfn = dfn.dropna()
    #Separate data into explanatory and response variables
    XCAT = dfn.filter(items = c).values
    XNUM = dfn.filter(items = d).values
    y = None
    #Encode cateogrical data and merge with numerical data
    onehotencoder = load('onehot.joblib')
    XCAT = onehotencoder.transform(XCAT).toarray()
    X = np.concatenate((XCAT, XNUM), axis=1)
    return X, dfn

'''
Runs the model that classifies each request as more than or less than/equal to 11 days. Parameters are the hyperparameters for the model itself, and the train data and labels.
'''
def estimation_model(estimators, depth,X,y):    
    rf = RandomForestClassifier(n_estimators = estimators, max_depth = depth)
    print('creating train, test, val split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
    #Train Model on 3 yrs data on
    print('training model')
    rf.fit(X_train, y_train)
    #Test model
    print('testing model')
    y_vpred = rf.predict(X_val)
    print('dump to pickle')
    pickle.dump(rf, open('randomForest.pkl', 'wb'))
    #Print Accuracy Function results
    print("Accuracy:",metrics.accuracy_score(y_val, y_vpred))
    print("Precision, Recall, F1Score:",metrics.precision_recall_fscore_support(y_val, y_vpred, average = 'binary'))
    return rf

'''
Takes a file path to the data, runs the appropriate preprocessing steps, and uses the model to classify everything into the majority and minority class. Returns a dataframe with the majority class and a separate one with the minority class.
'''
def split_to_models(df,formatted=False, encode=True):
    print('Calculating train data and labels')
    X, y, df = preprocess(df,formatted)
    print('Creating 11 day classifier')
    #Train on 3 yrs before data
    print(X.shape)
    model_eleven = estimation_model(50,20,X,y)
    df['LessEqualEleven'] = model_eleven.predict(X)
    df['LessEqualEleven'] = df['LessEqualEleven'].apply(lambda x: int(x))
    df_sgcrf = df[df['LessEqualEleven'] == 1.0]
    df_other = df[df['LessEqualEleven'] == 0.0]
    return df_sgcrf, df_other

'''
Given a start date, it will train the classifier on data 3 years before 10 weeks before the start date, then 
train the sgcrf model with 10 weeks of data from before the start date, then predict using the fifty most recent requests
from the start date.
'''

def create_models(start_date, request_type, CD):
    #Current date
    start = datetime.strptime(start_date,'%Y-%m-%d')
    con = psycopg2.connect(database="311_db", user="311_user", password="311_pass", host="localhost", port="5432")
    # The * will be replaced by the proper columns, then the preprocess function will be edited 
    df = pd.read_sql("SELECT * FROM requests WHERE createddate >= CURRENT_DATE - INTERVAL '3 years 11 weeks'",con)
    #Preprocess data
    X, y, dfn = preprocess(df,encode=True)
    #Sort into past three years from 11 weeks before
    df_three = dfn[(dfn['Just Date'] <= start-timedelta(weeks=11)) &
                   (dfn['Just Date'] >= start-timedelta(weeks=11)+relativedelta(years=-3) )]
    #Run dataframe through the classifier and get all requests less than or equal to 11 days
    df_sgcrf,ignore = split_to_models(df_three,True)
    #Date of the 50th request from the end
    train_end_date = df_sgcrf.iloc[-50]['Just Date']
    #Send to lagcrf
    modelCD, modelRT = lagcrf.crf_models(df_sgcrf.copy(),df_sgcrf.copy(), train_end_date - timedelta(weeks=10), 
        train_end_date, train_end_date - timedelta(weeks=10), train_end_date, request_type, CD, 50)
    #Dump to pickle file
    pickle.dump(modelCD, open('modelCD.pkl','wb'))
    pickle.dump(modelRT, open('modelRT.pkl','wb'))

