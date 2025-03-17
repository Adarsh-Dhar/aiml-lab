import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def load_data():
    # Load the dataset from a CSV file
    data = pd.read_csv("derm.csv")
    return data

def preprocess_data(data):
    # Drop any rows with missing values
    data = data.dropna()

    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert class labels to one-hot encoding
    y = to_categorical(y - 1)  # Assuming class labels start from 1

    return X, y

def split_data(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model(input_dim, output_dim, output_activation='sigmoid'):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation=output_activation))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Load and preprocess the data
    data = load_data()
    X, y = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train the model with sigmoid output activation
    model_sigmoid = build_model(X_train.shape[1], y_train.shape[1], output_activation='sigmoid')
    history_sigmoid = model_sigmoid.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

    # Build and train the model with ReLU output activation
    model_relu = build_model(X_train.shape[1], y_train.shape[1], output_activation='relu')
    history_relu = model_relu.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

    # Evaluate the models
    loss_sigmoid, accuracy_sigmoid = model_sigmoid.evaluate(X_test, y_test, verbose=0)
    loss_relu, accuracy_relu = model_relu.evaluate(X_test, y_test, verbose=0)

    print(f'Test Accuracy with sigmoid: {accuracy_sigmoid * 100:.2f}%')
    print(f'Test Accuracy with ReLU: {accuracy_relu * 100:.2f}%')

    # Plot training history for comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_sigmoid.history['loss'], label='sigmoid Loss')
    plt.plot(history_relu.history['loss'], label='ReLU Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history_sigmoid.history['accuracy'], label='sigmoid Accuracy')
    plt.plot(history_relu.history['accuracy'], label='ReLU Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
