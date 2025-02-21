# data_preprocessing.py

import pandas as pd

def preprocess_data(brent_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Brent oil price data.
    
    :param brent_data: DataFrame of Brent prices.
    :return: Cleaned DataFrame.
    """
    # Handle missing values using forward fill
    brent_data.ffill(inplace=True)  # Updated line to avoid the warning
    return brent_data





