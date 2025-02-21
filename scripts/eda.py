# scripts/eda.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_prices(df):
    """
    Plot the historical Brent crude oil prices.
    
    :param df: DataFrame containing the Brent prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Historical Brent Oil Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_acf_pacf(df):
    """
    Plot the ACF and PACF for the Brent crude oil prices.
    
    :param df: DataFrame containing the Brent prices.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['Price'], ax=ax[0])
    plot_pacf(df['Price'], ax=ax[1])
    
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.show()