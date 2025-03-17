import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def create_synthetic_data():
    # Create a synthetic dataset for visualization
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42)
    y = to_categorical(y)
    return X, y

def build_model(input_dim, output_dim, activation='relu'):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(output_dim, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_decision_boundaries(X, y, model, title):
    # Plot decision boundaries
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    # Create synthetic data
    X, y = create_synthetic_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train the model with ReLU activation
    model_relu = build_model(X_train.shape[1], y_train.shape[1], activation='relu')
    model_relu.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

    # Build and train the model with Tanh activation
    model_tanh = build_model(X_train.shape[1], y_train.shape[1], activation='tanh')
    model_tanh.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

    # Plot decision boundaries
    plot_decision_boundaries(X_train, y_train, model_relu, 'Decision Boundaries with ReLU Activation')
    plot_decision_boundaries(X_train, y_train, model_tanh, 'Decision Boundaries with Tanh Activation')

if __name__ == "__main__":
    main()
