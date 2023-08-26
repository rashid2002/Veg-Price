import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/hrithiksharma/Desktop/internproject/cleaned_data1.csv')

selected_attributes = ['min_price', 'max_price', 'modal_price']
data_selected = data[selected_attributes].values

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_selected)

time_steps = 20

X = []
y = []
for i in range(len(data_scaled) - time_steps):
    X.append(data_scaled[i:i+time_steps])
    y.append(data_scaled[i+time_steps])
X, y = np.array(X), np.array(y)

split_ratio = 0.8
split_idx = int(split_ratio * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, len(selected_attributes))))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(selected_attributes)))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions_actual = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train)
test_predictions_actual = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test)


vegetable_names = data['vegetable'][time_steps:].values

import streamlit as st

# Assuming train_predictions_actual and y_train_actual are 2D arrays
actual_train_prices = y_train_actual[:, 0]
predicted_train_prices = train_predictions_actual[:, 0]

# Plotting the actual and predicted prices using Streamlit's line_chart
import streamlit as st

# Assuming train_predictions_actual and y_train_actual are 2D arrays
actual_train_prices = y_train_actual[:, 0]
predicted_train_prices = train_predictions_actual[:, 0]

# Plotting the actual and predicted prices using Streamlit's line_chart -------
st.line_chart(pd.DataFrame({
    'Actual Train Prices': actual_train_prices,
    'Predicted Train Prices': predicted_train_prices
}), width=800, height=400)

# Display the vegetable names using a Streamlit table
st.table(pd.DataFrame({
    'Vegetable': vegetable_names[:split_idx]
}))


