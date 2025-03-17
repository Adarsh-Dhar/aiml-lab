import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def create_synthetic_data():
    # Create a synthetic dataset for soil types
    X, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42)
    y = to_categorical(y)
    return X, y

def preprocess_data(X, y):
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def build_model(input_dim, output_dim):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_training_history(history):
    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

def main():
    # Create synthetic data
    X, y = create_synthetic_data()

    # Preprocess the data
    X, y = preprocess_data(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1], y_train.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
