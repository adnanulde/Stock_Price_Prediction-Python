import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model 
import streamlit as st
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Load the model
model_path = 'C:/Users/adnan/Python/Jupyter Notebook/stock_prediction/Stock Prediction Model.keras'
model = load_model(model_path)

# Streamlit App
st.header('Stock Price Prediction')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
future_days = st.slider('Select Number of Future Days to Predict', min_value=1, max_value=30, value=7)
start = '2014-01-01'
end = '2024-08-30'

# Fetching Stock Data from Yahoo Finance
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error('Invalid stock symbol. Please try again.')
        st.stop()  # Stop execution if data is empty
except Exception as e:
    st.error(f'An error occurred: {e}')

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(data_train)
test_scaled = scaler.transform(data_test)

# Prepare input data for future predictions
past_100_days = data['Close'].tail(100).values.reshape(-1, 1)
scaled_past_100 = scaler.transform(past_100_days)

x_input = scaled_past_100[-100:].reshape(1, 100, 1)


# Generate Future Predictions
future_predictions = []
for _ in range(future_days):
    pred = model.predict(x_input, verbose=0)
    future_predictions.append(pred[0][0])
    
    # Reshape the predicted value to match input dimensions (1, 1, 1)
    pred_reshaped = np.array(pred[0][0]).reshape(1, 1, 1)
    
    # Update x_input by appending the predicted value
    x_input = np.append(x_input[:, 1:, :], pred_reshaped, axis=1)


# Rescale the predicted values back to original range
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a DataFrame for future dates and predicted prices
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Price'])

# Display Future Predictions
st.subheader('Future Price Predictions')
st.write(future_df)

# Prepare training and testing data
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_train_scale = scaler.fit_transform(data_test)

# Plot Moving Averages
st.subheader('Price VS Moving Avg. 50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price VS Moving Avg. 50 VS Moving Avg. 100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price VS Moving Avg. 100 VS Moving Avg. 200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i - 100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict prices
predict = model.predict(x)
scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

# Plot original vs predicted prices
st.subheader('Original Price VS Predicted Price')
fig4 = plt.figure(figsize=(10, 8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Plot Historical and Future Prices
st.subheader('Historical vs Future Price Prediction')
fig5 = plt.figure(figsize=(10, 8))
plt.plot(data['Close'], label='Historical Prices', color='blue')
plt.plot(future_df, label='Future Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)