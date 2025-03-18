import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def create_dataset(file_path='soil_data.csv'):
    # Generate synthetic dataset
    np.random.seed(42)
    num_samples = 500
    
    moisture_content = np.random.uniform(5, 40, num_samples)
    density = np.random.uniform(1.2, 2.5, num_samples)
    shear_strength = np.random.uniform(20, 100, num_samples)
    
    soil_types = np.random.choice(['Clay', 'Silt', 'Sand', 'Loam'], num_samples)
    
    data = pd.DataFrame({
        'Moisture_Content': moisture_content,
        'Density': density,
        'Shear_Strength': shear_strength,
        'Soil_Type': soil_types
    })
    
    data.to_csv(file_path, index=False)

def load_data(file_path='soil_data.csv'):
    # Load the dataset from a CSV file
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Separate features and target
    X = data.drop('Soil_Type', axis=1)
    y = data['Soil_Type']
    
    # Encode categorical target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)  # Convert labels to one-hot encoding
    
    return X, y, label_encoder

def build_model(input_dim, output_dim):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))  # Output layer for classification
    
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
    # Create synthetic dataset
    create_dataset()
    
    # Load and preprocess the data
    data = load_data()
    X, y, label_encoder = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train the model
    model = build_model(X_train.shape[1], y.shape[1])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
