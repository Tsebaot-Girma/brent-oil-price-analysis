from statsmodels.tsa.arima.model import ARIMA
#import numpy as np
#from keras.models import Sequential
#from keras.layers import LSTM, Dense
#from sklearn.preprocessing import MinMaxScaler


#from pmdarima import auto_arima
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

def fit_arima_model(df, order=(1, 1, 1)):
    """Fit an ARIMA model to the Brent crude oil prices."""
    model = ARIMA(df['Price'], order=order)
    model_fit = model.fit()
    return model_fit




def fit_lstm_model(brent_data):
    """Fit an LSTM model to the Brent crude oil prices."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(brent_data[['Price']].values)

    train_size = int(len(scaled_data) * 0.8)
    train, test = scaled_data[0:train_size], scaled_data[train_size:]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 10
    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, y_test