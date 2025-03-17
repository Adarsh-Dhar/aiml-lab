import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path='environmental_data.csv'):
    # Load the dataset from a CSV file
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column):
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
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

def main(file_path='environmental_data.csv', target_column='target'):
    # Load and preprocess the data
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
