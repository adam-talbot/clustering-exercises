####### WRANGLE MALL CUSTOMER DATA #######


# standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# custom imports
from env import host, user, password


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
def new_mall_data():
    '''
    This function reads the data from the Codeup db into a df and returns the df.
    '''
    # Create SQL query.
    sql_query = """
    select * from customers;
    """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    
    return df.set_index('customer_id')

def get_mall_data():
    '''
    This function reads in data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('mall_customers_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_mall_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('mall_customers_df.csv')
        
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
        
def remove_outliers(df):
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