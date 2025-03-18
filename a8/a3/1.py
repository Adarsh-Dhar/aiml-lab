import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path='water_quality.csv'):
    # Load the dataset from a CSV file
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Drop unnecessary columns
    drop_columns = ['Site_Id', 'Unit_Id', 'Read_Date', 'Time (24:00)', 'Field_Tech', 'DateVerified', 'WhoVerified', 'Year']
    data = data.drop(columns=drop_columns, errors='ignore')
    
    # Handle missing values (fill with mean)
    data = data.fillna(data.mean())
    
    # Define target variable (WQI calculation)
    def calculate_wqi(row):
        weights = {
            'pH (standard units)': 0.2,
            'Dissolved Oxygen (mg/L)': 0.3,
            'Salinity (ppt)': 0.1,
            'Secchi Depth (m)': 0.15,
            'Water Depth (m)': 0.1,
            'Water Temp (?C)': 0.15
        }
        return sum(row[param] * weights.get(param, 0) for param in weights.keys())
    
    data['WQI'] = data.apply(calculate_wqi, axis=1)
    
    # Separate features and target
    X = data.drop(columns=['WQI'])
    y = data['WQI']
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
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Training and Validation MAE')
    plt.show()

def main():
    # Load and preprocess the data
    data = load_data()
    X, y = preprocess_data(data)

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
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Mean Absolute Error: {mae:.2f}')

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()