
# 🟡 Gold Price Prediction with LSTM

This project demonstrates how to build a **machine learning model** that predicts gold prices using historical data and a Long Short-Term Memory (LSTM) neural network.
It downloads gold price data from **Yahoo Finance**, preprocesses it, trains an LSTM model, evaluates performance, and forecasts the next 5 days.

---

## 📌 Features

* Fetches historical **gold futures prices (GC=F)** from Yahoo Finance.
* Preprocesses data with:

  * Scaling using **MinMaxScaler**.
  * Sequence creation (last 60 days → predict next day).
* Builds and trains a **2-layer LSTM model** in TensorFlow/Keras.
* Evaluates predictions on a test set with **Root Mean Squared Error (RMSE)**.
* Plots actual vs predicted gold prices for visualization.
* Forecasts **the next 5 days of gold prices**.

---

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install yfinance pandas scikit-learn tensorflow matplotlib
```

---

## 📂 Project Workflow

### 1. **Data Collection**

* Uses `yfinance` to download daily gold futures data (`GC=F`) from 2010–2025.

```python
symbol = "GC=F"  # Gold continuous futures price
gold_data = yf.download(symbol, start="2010-01-01", end="2025-01-01")
```

### 2. **Data Preprocessing**

* Extracts the **Close** price.
* Scales values to `[0,1]` with `MinMaxScaler`.
* Splits into training (80%) and testing (20%) sets using sequential split (not random, since time-series data must keep order).
* Converts data into **60-day sequences** to predict the next day.

```python
X_train, y_train = create_sequences(train_data, n_steps=60)
```

### 3. **Model Building**

* A sequential **LSTM model**:

  * 1st LSTM layer with 50 units (`return_sequences=True`).
  * 2nd LSTM layer with 50 units.
  * Dense output layer for predicting the next price.

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
```

### 4. **Training**

* Trains for 20 epochs with a batch size of 32.
* Uses **MSE loss** and **Adam optimizer**.
* Validates performance on the test set.

### 5. **Evaluation**

* Predicts test set prices, inverse-transforms them back to USD.
* Computes **RMSE** to measure prediction error.
* Plots **Actual vs Predicted Prices** with matplotlib.

![Example Plot](https://matplotlib.org/stable/_images/sphx_glr_plot_001.png)
*(Illustrative example — your plot will show gold price curves.)*

### 6. **Forecasting Next 5 Days**

* Starts with the last 60 days of data.
* Iteratively predicts 1 day at a time, sliding the window forward.
* Outputs predicted gold prices in USD.

```bash
Predicted gold prices for the next 5 days:
Day 1: $1850.23
Day 2: $1847.95
...
```

---

## 📊 Example Output

```
Test RMSE: 20.45 USD

Predicted gold prices for the next 5 days:
Day 1: $1850.23
Day 2: $1847.95
Day 3: $1852.67
Day 4: $1855.10
Day 5: $1858.42
```

---

## 🚀 Next Steps

* Add **technical indicators** (moving averages, RSI, volatility) as features.
* Tune hyperparameters (LSTM units, epochs, batch size).
* Try alternative models (e.g., **XGBoost**, **Transformers**) for comparison.
* Deploy predictions via a web dashboard (e.g., Streamlit, FastAPI).

---

## 📚 References

* [TensorFlow LSTM Docs](https://www.tensorflow.org/guide/keras/rnn)
* [yfinance Documentation](https://pypi.org/project/yfinance/)
* [Gold Price LSTM Prediction Tutorial](https://medium.com/@rajat404/stock-price-prediction-using-lstm-86e4d94a06c8)

---

👉 This project is intended for **educational purposes only**. Predicting financial markets is inherently uncertain — use with caution.

---

