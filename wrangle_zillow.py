import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
import acquire_zillow as a
import prepare_zillow as p

####### WRANGLE ZILLOW MODULE #######

### ACQUIRE ###

def get_zillow_data():
    '''
    This function reads in data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    return df = a.get_zillow_data()

### PREPARE ###

def clean_zillow(df):
    '''
    Take in df and eliminates all database key columns, further refines to only single unit properties, handles all nulls with various methods
    '''
    return df = p.clean_zillow(df)

def split(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 60/20/20 for train/validate/test
    '''
    return train, validate, test = p.split(df)

def encode_scale(df, scaler):
    '''
    Takes in df and scaler of your choosing and returns scaled df with unscaled columns dropped
    '''
    return train, validate, test = p.encode_scale(df, scaler)