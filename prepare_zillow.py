####### PREPARE ZILLOW DATA #######

# standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    df = df.drop(columns=[col for col in df.columns.tolist() if col.endswith('id')]) # remove database table keys
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Mobile Home') \
         | (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes')] # my way to filter for single unit properties
    df = handle_missing_values(df) # remove columns based on thresholds for nulls
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median()) # impute sf using median
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median()) # impute lot sf using median
    df.propertyzoningdesc = df.propertyzoningdesc.fillna(df.propertyzoningdesc.mode().tolist()[0]) # impute zoning code using most commonly occuring code
    df.regionidcity = df.regionidcity.fillna(df.regionidcity.mode().tolist()[0]) # impute city id using most commonly occuring id
    df.unitcnt = df.unitcnt.fillna(1) # impute with 1 since prop types were chosen to target this
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('Central') # impute with most commonly occuring type
    df = df.dropna() # drop any remaining nulls
    df['county'] = df['fips'].apply(lambda x: 'Los Angeles' if x == 6037 else 'Orange' if x == 6059 else 'Ventura') # convert to county names
    df.censustractandblock = '0' + df.censustractandblock.astype('int').astype('string').astype('object') # format back to object and recover leading 0
    df['zip'] = df.regionidzip.astype(int).astype('object') # convert to objectregionidzip
    df = df.drop(columns=['calculatedbathnbr', # all present values are same as beds + baths, redundant
                          'finishedsquarefeet12', # all present values are same as other sf column, redundant
                          'fullbathcnt', # redundant info, all necessary info is contained in bathroomcnt
                          'rawcensustractandblock', # another column containted identical info, redundant
                          'fips', # I converted this to county name, no longer needed
                          'regionidcity', # have more geographically-specific info, couldn't find code meaning/translation
                          'regionidcounty', # already have county info in fips, couldn't find code meaning/translation
                          'regionidzip', # created new, reformatted column
                          'assessmentyear', # only one value present in data, not helpful
                          'landtaxvaluedollarcnt', # info contained in total assessed value (land + structure)
                          'structuretaxvaluedollarcnt', # info contained in total assessed value (land + structure)
                          'taxamount', # derived from tax assessed value, redundant
                          'unitcnt', # all info in column would be 1, not helpful
                         ]
                )
    unit_type_dict = { # created dictionary for all unit type conversions
    'bedroomcnt' : 'int',
    'calculatedfinishedsquarefeet' : 'int',
    'latitude' : 'int',
    'longitude' : 'int',
    'lotsizesquarefeet' : 'int',
    'roomcnt' : 'int',
    'yearbuilt' : 'int',
    'transactiondate' : 'datetime64'
                    }
    df = df.astype(unit_type_dict) # convert unit types
    df.transactiondate = df.transactiondate.dt.date # remove time from time date format, not needed
    cols_w_outliers = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt'] # of remaining columns, these need outliers removed
    df = df = remove_outliers(df, cols_w_outliers) # call function to remove outliers using Tukey method
    room_cnt_ratio = 0.8333333333333334 # ratio of beds + baths to roomcnt present in data
    df.roomcnt[df.roomcnt < (df.bedroomcnt + df.bathroomcnt)] = round(df.bathroomcnt + df.bedroomcnt / room_cnt_ratio, 0).astype('int') # impute impossible roomcnt values using ratio from rest of dataset
    rename_dict = {
    'transactiondate' : 'sale_date',
    'bathroomcnt' : 'bath_cnt',
    'bedroomcnt' : 'bed_cnt',
    'calculatedfinishedsquarefeet' : 'sqft',
    'lotsizesquarefeet' : 'lot_sqft',
    'propertycountylandusecode' : 'land_use_code',
    'propertyzoningdesc' : 'zoning_desc',
    'roomcnt' : 'total_rooms',
    'yearbuilt' : 'year_built',
    'taxvaluedollarcnt' : 'assessed_value',
    'censustractandblock' : 'census_tract_block',
    'logerror' : 'log_error',
    'heatingorsystemdesc' : 'heating_system',
    'propertylandusedesc' : 'land_use'
    }
    df = df.rename(columns=rename_dict)
    return df

def nulls_by_col(df):
    '''
    Takes in df and shows count of how many rows are null and percentage of total rows that are null
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = round(num_missing / rows * 100, 2)
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    '''
    Takes in df and shows count of how many columns are null and percentage of total columns that are null and value count of each unique combo
    '''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = round(num_missing / df.shape[1] * 100, 2)
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(columns={'index': 'num_rows'}).reset_index()
    return rows_missing
    
def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    Takes in df and thresholds for null proportions in each column and row and returns df with only columns and rows below threshold
    '''
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df
        
def remove_outliers(df, cols):
    '''
    Removes outliers that are outside of 1.5*IQR
    '''
    for col in cols:
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