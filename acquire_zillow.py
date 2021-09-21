import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split

##################### Acquire Zillow Data #####################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_sfr_data():
    '''
    This function reads the data from the Codeup db into a df and returns the df.
    '''
    # Create SQL query.
    sql_query = """
    select *
    from properties_2017 as prop
    join(
        select parcelid, max(transactiondate) as transactiondate
        from predictions_2017
        group by parcelid
         ) as txn using(parcelid)
    join predictions_2017 as pred using(parcelid, transactiondate)
    left join airconditioningtype as act using (airconditioningtypeid)
    left join architecturalstyletype as ast using(architecturalstyletypeid)
    left join buildingclasstype as bct using(buildingclasstypeid)
    left join heatingorsystemtype as hst using(heatingorsystemtypeid)
    left join propertylandusetype as plt using(propertylandusetypeid)
    left join storytype as st using(storytypeid)
    left join typeconstructiontype as tct using(typeconstructiontypeid)
    where latitude IS NOT NULL and longitude IS NOT NULL; 
    ;
    """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_sfr_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_sfr_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_sfr_data()
        
        # Write DataFrame to a csv file.
        # df.to_csv('zillow_sfr_df.csv')
        
    return df
