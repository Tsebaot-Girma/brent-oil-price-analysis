# data_preprocessing.py


import pandas as pd

def preprocess_data(brent_data):
    # Handle missing values
    brent_data.fillna(method='ffill', inplace=True)

    #Create moving average for smoothing
    brent_data['Moving_Average'] = brent_data['Price'].rolling(window=30).mean()
    
    return brent_data

