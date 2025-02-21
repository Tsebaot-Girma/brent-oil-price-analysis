# scripts/modeling.py

from statsmodels.tsa.arima.model import ARIMA

def fit_arima_model(df, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the Brent crude oil prices.
    
    :param df: DataFrame containing the Brent prices.
    :param order: Tuple representing the ARIMA order (p, d, q).
    :return: Fitted ARIMA model.
    """
    model = ARIMA(df['Price'], order=order)
    model_fit = model.fit()
    return model_fit


