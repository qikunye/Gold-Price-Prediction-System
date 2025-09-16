import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Define the ticker and date range for gold prices
symbol = "GC=F"  # Gold continuous futures price
start_date = "2010-01-01"
end_date   = "2025-01-01"

# Download historical daily data for the given period
gold_data = yf.download(symbol, start=start_date, end=end_date)

# Extract the 'Close' prices as our target series
closing_prices = gold_data['Close'].values  # NumPy array of prices
closing_prices = closing_prices.reshape(-1, 1)  # reshape to column vector



scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(closing_prices)


# randomly shuffles the data before splitting - from sklearn.model_selection import train_test_split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Manual sequential split (time-series style) - Use 80% of data for training, rest for testing
training_size = int(len(closing_prices_scaled) * 0.8)
train_data = closing_prices_scaled[:training_size]
test_data  = closing_prices_scaled[training_size - 60:]  


##hiii odjwpo'dqw'dkqwkqwskwdlqwkd
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, 0])       # sequence of length n_steps
        y.append(data[i+n_steps, 0])         # the next value after the sequence
    X = np.array(X)
    y = np.array(y)
    return X, y

# Prepare sequences with a chosen window size (e.g., 60 days)
n_steps = 60
X_train, y_train = create_sequences(train_data, n_steps)
X_test,  y_test  = create_sequences(test_data, n_steps)

# Reshape X arrays to 3D: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(units=50))  # second LSTM layer (return_sequences=False by default)
model.add(Dense(units=1))  # output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the LSTM model on the training set
history = model.fit(X_train, y_train, 
                    epochs=20, batch_size=32, 
                    validation_data=(X_test, y_test))




# Predict on the test set
y_pred_scaled = model.predict(X_test)
# Inverse transform the scaled predictions and true values back to original scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE on test data
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
print(f"Test RMSE: {rmse:.2f} USD")




# Plot actual vs predicted prices for the test set
plt.figure(figsize=(10,6))
plt.plot(y_true, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.title('Gold Price Prediction on Test Data')
plt.xlabel('Time (days)')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.show()

# Number of days to forecast
days_to_predict = 5
# Start with the last available data as the initial sequence
last_sequence = closing_prices_scaled[-n_steps:]  # last 60 days from the entire dataset
last_sequence = last_sequence.reshape(1, n_steps, 1)  # reshape to model input shape

predictions_scaled = []
for i in range(days_to_predict):
    next_pred_scaled = model.predict(last_sequence)  # predict next day (scaled)
    predictions_scaled.append(next_pred_scaled[0, 0])
    # Update the sequence: remove the first value and append the new prediction
    next_input = np.append(last_sequence[:,1:,:], [[next_pred_scaled[0,0]]], axis=1)
    last_sequence = next_input  # set this as the new input sequence

# Convert predictions back to original scale
predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions_scaled)

print("Predicted gold prices for the next 5 days:")
for i, price in enumerate(predictions, 1):
    print(f"Day {i}: ${price[0]:.2f}")

