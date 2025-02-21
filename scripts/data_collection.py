# data_collection.py

import pandas as pd

def fetch_brent_data(file_path: str) -> pd.DataFrame:
    """
    Load historical Brent crude oil prices from a CSV file.
    
    :param file_path: Path to the CSV file containing Brent prices.
    :return: DataFrame containing the Brent prices.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Modify the date parsing to match the actual format
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)  # Adjusted format
        df.set_index('Date', inplace=True)
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")