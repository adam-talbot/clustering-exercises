####### PREPARE ZILLOW DATA #######

# standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

def plot_distributions(df):
    '''
    This function creates frequency distributions for each numerical column in the df
    '''
    plt.figure(figsize=(15, 3))
    
    # List of columns
    cols = df.select_dtypes('number').columns.tolist()
    
    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,len(cols), plot_number)
        
        # Axis labels
        plt.xlabel(col)
        plt.ylabel('count')
        
        # Display histogram for column.
        df[col].hist(edgecolor='black', color='green')
        
        # Hide gridlines.
        plt.grid(False)
        
def plot_boxplots(df):
    '''
    This function creates boxplots for each numerical column in the df
    '''
    plt.figure(figsize=(12, 4))

    # List of columns
    cols = df.select_dtypes('number').columns.tolist()
    
    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,len(cols), plot_number)
        
        # Title with column name.
        #plt.title(col)
        
        # Display boxplot for column.
        sns.boxplot(y=col, data=df, color='green')
        plt.tight_layout()
        
def clean_zillow(df):
    '''
    Take in df and eliminates all database key columns, further refines to only single unit properties, handles all nulls with various methods
    '''
    df = df.drop(columns=[col for col in df.columns.tolist() if col.endswith('id')])
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Mobile Home') \
         | (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes')]
    df = handle_missing_values(df)
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median())
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median())
    df.propertyzoningdesc = df.propertyzoningdesc.fillna(df.propertyzoningdesc.mode().tolist()[0])
    df.regionidcity = df.regionidcity.fillna(df.regionidcity.mode().tolist()[0])
    df.unitcnt = df.unitcnt.fillna(1)
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('Central')
    df = df.dropna()
    df = df.drop(columns=['calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt'])
    return df
    
def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    Takes in df and thresholds for null proportions in each column and row and returns df with only columns and rows below threshold
    '''
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df
        
def remove_outliers(df):
    '''
    Removes outliers that are outside of 1.5*IQR
    '''
    num_cols = df.select_dtypes('number').columns.tolist()
    for col in num_cols:
        Q1 = np.percentile(df[col], 25, interpolation='midpoint')
        Q3 = np.percentile(df[col], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        UB = Q3 + (1.5 * IQR)
        LB = Q1 - (1.5 * IQR)
        df = df[(df[col] < UB) & (df[col] > LB)]
    return df

def split(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 60/20/20 for train/validate/test
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=527)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=527)
    return train, validate, test

def encode_scale(df, scaler):
    '''
    Takes in df and scaler of your choosing and returns scaled df with unscaled columns dropped
    '''
    cat_cols = df.select_dtypes('object').columns.tolist()
    df = pd.get_dummies(data=df, columns=cat_cols)
    train, validate, test = split(df)
    num_cols = df.select_dtypes('int64').columns.tolist()
    new_column_names = [c + '_scaled' for c in num_cols]
    
    # Fit the scaler on the train
    scaler.fit(train[num_cols])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[num_cols]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[num_cols]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[num_cols]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    # drop scaled columns
    train = train.drop(columns=num_cols)
    validate = validate.drop(columns=num_cols)
    test = test.drop(columns=num_cols)
    
    return train, validate, test