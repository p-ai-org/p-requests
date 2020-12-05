import pandas as pd
import numpy as np 
from datetime import datetime
from datetime import date
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sgcrf import SparseGaussianCRF
 

def elapsedHours(createdDate, closedDate): 
    """
    - Returns elapsedHours between two dates
    """
    
    #Convert date/time strings to datetime objects
    created = datetime.strptime(createdDate, '%m/%d/%Y %I:%M:%S %p')
    closed = datetime.strptime(closedDate, '%m/%d/%Y %I:%M:%S %p')
    
    #Difference between createdDate and closedDate
    diff = closed - created
    
    #Compute elapsedHours
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    minutes = minutes + seconds/60
    hours = hours + minutes/60

    return hours


def CreateTrainSet(df, predictor_num):
    """
    - Creates training set.
    - Accepts df with training data.
    - predictor_num is the specified number of previous elapsedHour values used to predict 
    the next one.   
    """
    X_train = []
    X_test = []
    
    #Create training set 
    for i in range(0,df.shape[0] - predictor_num): 
        X_train.append(df['ElapsedHours'][i:i + predictor_num].values)
        X_test.append(df['ElapsedHours'][i + predictor_num])
    
    return np.asarray(X_train), np.asarray(X_test)
    

def CreateTestSet(df, predictor_num):
    """
    - Similar to CreateTrainSet. Creates testing set.
    - Accepts df with testing data.
    - predictor_num is the specified number of previous elapsedHour values used to predict 
    the next one. Same value as before   
    """
    y_train = []
    y_test = []
    
    #Create testing Set 
    for i in range(0,df.shape[0] - predictor_num): 
        y_train.append(df['ElapsedHours'][i:i + predictor_num].values)
        y_test.append(df['ElapsedHours'][i + predictor_num])

    return np.asarray(y_train), np.asarray(y_test)

def metrics(y_hat,y):
    """
    Return rmse and mae
    """
    rmse_days = np.sqrt(mean_squared_error(y_hat, y, multioutput='raw_values'))
    mae = mean_absolute_error(y_hat, y)
    
    return rmse_days, mae


def CleanedFrame(df): 
    """
    Creates elapsedHours column, and converts
    CreatedDate and ClosedDate columns to datetime objects, Sorts dataframe 
    chronologically by CreatedDate
    """
    df['ElapsedHours'] = df.apply(lambda x: elapsedHours(x['CreatedDate'],x['ClosedDate']),axis=1)
    df['ClosedDate'] = df['ClosedDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
    df['CreatedDate'] = df['CreatedDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
    
    df = df.sort_values(by="CreatedDate")

    return df


def preprocessing(df, start_date, end_date):
    """
    Filters dataframe by specified start and end_dates and runs CleanedFrame on it.  
    """ 

    #Filter dataframe by dates 
    df['Just Date'] = df['Just Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df = df[(df['Just Date'] >= start_date) & (df['Just Date'] <= end_date)]
    df = CleanedFrame(df)

    return df

#Model
def lacer(df, df1, train_start_date, train_end_date, test_start_date, test_end_date, request_type, owner, CD, predictor_num): #Once model is ready, replace df with csv
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
    y_train, y_test = CreateTestSet(dftest, predictor_num)
    y_test = y_test.reshape((-1, 1))


## 3 Models
    #Model1: CD
    modelCD = SparseGaussianCRF(lamL=0.1, lamT=0.1, n_iter=10000)
    dftrainCD = dftrain[dftrain['CD'] == CD].reset_index(drop = True)

    X_trainCD, X_testCD = CreateTrainSet(dftrainCD, predictor_num)
    X_testCD = X_testCD.reshape((-1, 1))
    modelCD.fit(X_trainCD, X_testCD)

    y_predCD = modelCD.predict(y_train)

    #Model2: Request_type
    modelRT = SparseGaussianCRF(lamL=0.1, lamT=0.1, n_iter=10000)
    dftrainRT = dftrain[dftrain['RequestType'] == request_type].reset_index(drop = True)

    X_trainRT, X_testRT = CreateTrainSet(dftrainRT, predictor_num)
    X_testRT = X_testRT.reshape((-1, 1))

    modelRT.fit(X_trainRT, X_testRT)

    y_predRT = modelRT.predict(y_train)

    #Model3: Owner
    modelOwner = SparseGaussianCRF(lamL=0.1, lamT=0.1, n_iter=10000)
    dftrainOwner = dftrain[dftrain['Owner'] == owner].reset_index(drop = True)

    X_trainOwner, X_testOwner = CreateTrainSet(dftrainOwner, predictor_num)
    X_testOwner = X_testOwner.reshape((-1, 1))

    modelOwner.fit(X_trainOwner, X_testOwner)

    y_predOwner = modelOwner.predict(y_train)

#Average out all predictions
    y_predFinal = (y_predCD + y_predRT + y_predOwner)/3

#Return metrics 
    return metrics(y_predFinal, y_test)




    