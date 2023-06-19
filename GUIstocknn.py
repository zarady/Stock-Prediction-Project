import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Main application window
window = tk.Tk()
window.title("Stock Market Prediction")
window.geometry("400x200")

# Label and entry for the CSV file path
file_label = tk.Label(window, text="CSV File Path:")
file_label.pack()
file_entry = tk.Entry(window)
file_entry.pack()

# A function to perform the prediction and display the results
def predict_stock_market():
    # to read the CSV file
    file_path = file_entry.get()
    try:
        stock_data = pd.read_csv('stock_data.csv')

        # Preprocess the data
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Date'] = stock_data['Date'].dt.date
        stock_data = stock_data[['Date', 'Close']]
        stock_data = stock_data.dropna() # Remove any missing values

        # Prepare the training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(np.array(stock_data['Close']).reshape(-1, 1))

        X = []
        y = []

        # Input sequences for time series prediction
        for i in range(30, len(stock_data)):
            X.append(scaled_close[i-30:i, 0])
            y.append(scaled_close[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Create and train the neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Predict the stock market for the next 30 days
        last_30_days = scaled_close[-30:]
        next_30_days = []
        for i in range(30):
            prediction = model.predict(last_30_days.reshape(1, -1))
            next_30_days.append(prediction)
            last_30_days = np.append(last_30_days[1:], prediction)

        next_30_days = scaler.inverse_transform(np.array(next_30_days).reshape(-1, 1))

        # Display the predictions
        next_30_days_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=30, freq='D')
        predictions_df = pd.DataFrame({'Date': next_30_days_dates, 'Predicted Close': next_30_days.flatten()})
        print(predictions_df)

        # Plot the predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Actual')
        plt.plot(predictions_df['Date'], predictions_df['Predicted Close'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Actual vs Predicted Stock Market')
        plt.legend()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# A button to initiate the prediction process
predict_button = tk.Button(window, text="Predict", command=predict_stock_market)
predict_button.pack()

# The GUI event loop
window.mainloop()