# LSTM Neural Network 

# This Stock Price Prediction program uses an artificial recurrent neural network called Long Short Term Memory
# Aims to predict the closing stock price of a corporation using the past x day stock prices

# Import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from tkinter import *
plt.style.use('fivethirtyeight')

stock_ticker = input("Enter Stock Ticker: ")
observed_interval = int(input("Enter Observed Interval: "))

# Obtain Stock Quote from yahoo Finance
df = web.DataReader(stock_ticker, data_source='yahoo', start='2012-01-01', end='2022-11-05')

# Visualise closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD ($)', fontsize=18)
#plt.show()

# Create new dataframe wwith only 'Close' column
data = df.filter(['Close'])
# Convert dataframe to numpy array
dataset = data.values
# Obtain number of rows to train model on (80%)
training_data_len = math.ceil(len(dataset) * .8)

# Scaling the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create scaled training data set
train_data = scaled_data[0:training_data_len , :]
# Split data into x_train and y_train data sets
x_train = []
y_train = []

for i in range (observed_interval, len(train_data)):
    x_train.append(train_data[i-observed_interval:i, 0])
    y_train.append(train_data[i, 0])
    
# Convert x_train & y_train
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model Architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25)) # 25 Neurons
model.add(Dense(1)) # 1 Neuron

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create Testing Dataset
test_data = scaled_data[training_data_len - observed_interval: , :]
# Create x_test & y_test 
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(observed_interval, len(test_data)):
    x_test.append(test_data[i-observed_interval:i, 0])
    
# Convert dataset into numpy array
x_test = np.array(x_test)

# Reshape data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Obtain model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# RMSE calculation
rmse = np.sqrt(np.mean((predictions- y_test)**2))
print(rmse)

# Plotting Data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualise Data
prediction_fig = plt.figure(figsize=(16,8))
plt.title(stock_ticker)
plt.xlabel('Data', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Export plot to plotly
prediction_plotly = py.plot_mpl(prediction_fig, filename= stock_ticker + "Prediction Graph")

# Save as HTML file
plotly.offline.plot(prediction_plotly, filename='Prediction Graph HTML', config={'displayModeBar': False})




