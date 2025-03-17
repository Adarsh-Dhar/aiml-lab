import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_synthetic_data():
    # Create a synthetic dataset for water quality parameters
    X, y = make_regression(n_samples=500, n_features=5, n_informative=5, n_targets=1, random_state=42)
    # Assuming y is the Water Quality Index (WQI)
    return X, y

def preprocess_data(X, y):
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def build_model(input_dim):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

def plot_training_history(history):
    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Training and Validation MAE')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Training and Validation MSE')

    plt.tight_layout()
    plt.show()

def main():
    # Create synthetic data
    X, y = create_synthetic_data()

    # Preprocess the data
    X, y = preprocess_data(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Mean Absolute Error: {mae:.2f}')
    print(f'Test Mean Squared Error: {mse:.2f}')

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
