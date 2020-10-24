import pandas as pd
import numpy as np 
from datetime import datetime
from datetime import date


###Data: https://data.lacity.org/A-Well-Run-City/MyLA311-Service-Request-Data-2020/rq3b-xjk8 ###


def convertTime(string):
    """Converts a string to a datetime object.
    """
    time = datetime.strptime(string,'%m/%d/%Y %I:%M:%S %p') 
    return time

def convertDays(string):
    """
    Converts string to date object. 
    """
    time = datetime.strptime(string[:10],'%m/%d/%Y') #Accepts time string, converts it to datetime object.
    return time

def convertFromSeconds(s): # total seconds
    """ convertFromSeconds(s): Converts an integer # of seconds into a list of [days, hours, minutes, seconds]
        input s: an int
    """
    s = s*60
    days = s // (24*60*60)  # total days
    s = s % (24*60*60) # remainder s
    hours = s // (60*60) # total hours
    s = s % (60*60) # remainder s
    minutes = s // 60 # total minutes
    s = s % 60 # remainder s
    statement = (days, ' days') + (hours, ' hrs') +(minutes, ' mins') + (s, 'sec')
    return statement
    
def elapsedTime(df):
    """
    Creates elapsedTime, elapsedDays, and date columns. 
    """
    #df = pd.read_csv(csv)
    
    hdf = df.dropna(axis=0, subset=['CreatedDate', 'ClosedDate'])

    #ElapsedTime 
    df1 = hdf['ClosedDate'].apply(convertTime, 0)   
    df2 = hdf['CreatedDate'].apply(convertTime, 0)
 
    hdf['ElapsedTime'] = df1 - df2
    hdf['ElapsedTime'] = hdf['ElapsedTime']/np.timedelta64(1,'m') 
    hdf['ElapsedTime'] = hdf['ElapsedTime'].apply(convertFromSeconds, 0)   
    
    #ElapsedDays
    df3 = hdf['CreatedDate'].apply(convertDays, 0)
    df4 = hdf['ClosedDate'].apply(convertDays, 0) 
    hdf['ElapsedDays'] = (df4 - df3).dt.days
    
    #Column for Closed Dates
    hdf['Just Date'] = df3
    
    return hdf.reset_index(drop = True).sort_index(axis=1)


def pipeline(csv): 
    """
    Take in 2020 dataset and return modified csv with ElapsedTime, ElapsedDays, and date columns
    Data: https://data.lacity.org/A-Well-Run-City/MyLA311-Service-Request-Data-2020/rq3b-xjk8
    """
    df = pd.read_csv(csv) 
    df = df.drop(143589,axis=0).reset_index(drop = True)  #This row was mislabled (the year inputted was 3020, not 2020)
    edf = elapsedTime(df)

    return edf.to_csv(r"C:\Users\hanaa\Downloads\cleaned_MyLA311_Service_Request_Data_2020.csv", index = False) #Pick name for the csv. 


def cleaned_df(csv): 
    """
    Take in 2020 dataset and return dataset with ElapsedTime, ElapsedDays, and date columns
    Data: https://data.lacity.org/A-Well-Run-City/MyLA311-Service-Request-Data-2020/rq3b-xjk8
    """
    df = pd.read_csv(csv) 
    df = df.drop(143589,axis=0).reset_index(drop = True)  #This row was mislabled (the year inputted was 3020, not 2020)
    edf = elapsedTime(df)
    edf = edf[edf.ElapsedDays >= 0]
    
    return edf