# ml/pipeline.py
import os
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_and_train_model(x_train, y_train, epochs=15, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def run_pipeline(symbol, start="2018-01-01", end=None, lookback=60, train_split=0.8, epochs=15, future_days=7):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # 1) Fetch data
    df = yf.download(symbol, start=start, end=end)
    if df.shape[0] < lookback + 5:
        raise ValueError("Not enough data for this symbol/date range.")
    close_prices = df[['Close']].copy()

    # 2) Scale
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_prices.values)

    # 3) Train/test split
    train_size = int(len(scaled) * train_split)
    train_data = scaled[:train_size]

    x_train, y_train = [], []
    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i-lookback:i, 0])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train).reshape(-1, lookback, 1)
    y_train = np.array(y_train)

    # 4) Load cached model or train new
    model_filename = f"{MODEL_DIR}/{symbol.replace('.', '_')}_lb{lookback}_ep{epochs}.h5"
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        model = build_and_train_model(x_train, y_train, epochs=epochs)
        model.save(model_filename)

    # 5) Test data
    test_data = scaled[train_size - lookback:]
    x_test, y_test = [], []
    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i-lookback:i, 0])
        y_test.append(test_data[i, 0])
    x_test = np.array(x_test).reshape(-1, lookback, 1)
    y_test = np.array(y_test)

    preds = model.predict(x_test)
    preds = scaler.inverse_transform(preds)
    real = scaler.inverse_transform(y_test.reshape(-1,1))

    # 6) Forecast future
    future_input = scaled[-lookback:].reshape(1, lookback, 1)
    future_preds_scaled = []
    for _ in range(future_days):
        p = model.predict(future_input)[0][0]
        future_preds_scaled.append(p)
        future_input = np.append(future_input[:,1:,:], [[[p]]], axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1))

    # 7) Dates
    dates = close_prices.index
    full_real = close_prices.values.flatten()
    preds_full = np.empty_like(full_real)
    preds_full[:] = np.nan
    preds_full[train_size:train_size + len(preds)] = preds.flatten()

    last_date = dates[-1]
    # Future dates (business days only)
    future_dates = pd.bdate_range(last_date, periods=future_days+1)[1:]


    return {
        "symbol": symbol,
        "dates": dates,
        "real": full_real,
        "preds_aligned": preds_full,
        "future_dates": future_dates,
        "future_preds": future_preds.flatten(),
        "model_file": model_filename
    }
