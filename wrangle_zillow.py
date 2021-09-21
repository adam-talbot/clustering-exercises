import acquire_zillow as a
import prepare_zillow as p

####### WRANGLE ZILLOW MODULE #######

### ACQUIRE ###

def get_zillow_data():
    '''
    This function reads in data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    df = a.get_zillow_data()
    return df

### PREPARE ###

def clean_zillow(df):
    '''
    Take in df and eliminates all database key columns, further refines to only single unit properties, handles all nulls with various methods
    '''
    df = p.clean_zillow(df)
    return df

def split(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 80/10/10 for train/validate/test
    '''
    train, validate, test = p.split_80(df)
    return train, validate, test

def encode_scale(df, scaler):
    '''
    Takes in df and scaler of your choosing and returns scaled df with unscaled columns dropped
    '''
    train, validate, test = p.encode_scale(df, scaler)
    return train, validate, test