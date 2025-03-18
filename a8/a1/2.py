import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path='concrete_data.csv'):
    # Load the dataset from a CSV file
    # Based on the data structure shown, defining column names
    column_names = [
        'cement', 
        'blast_furnace_slag', 
        'fly_ash', 
        'water', 
        'superplasticizer', 
        'coarse_aggregate', 
        'fine_aggregate', 
        'age', 
        'csMPa'
    ]
    
    try:
        # Try with predefined headers first
        data = pd.read_csv(file_path)
        # Check if headers need to be added
        if 'csMPa' not in data.columns:
            data = pd.read_csv(file_path, header=None, names=column_names)
    except:
        # If error occurs, try reading without headers
        data = pd.read_csv(file_path, header=None, names=column_names)
    
    return data

def preprocess_data(data):
    # Separate features and target
    X = data.drop('csMPa', axis=1)
    y = data['csMPa']
    return X, y

def build_model(input_dim):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def plot_training_history(history):
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Training and Validation MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Also plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess the data
    data = load_data()
    
    # Display basic information about the dataset
    print("Dataset Info:")
    print(f"Shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nSummary statistics:")
    print(data.describe())
    
    X, y = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the model
    model = build_model(X_train_scaled.shape[1])
    print("\nModel summary:")
    model.summary()
    
    # Train with more verbose output and early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=100, 
        batch_size=8, 
        validation_split=0.2, 
        verbose=1,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f'\nTest Mean Absolute Error: {mae:.2f}')
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('Actual vs Predicted Concrete Strength')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()