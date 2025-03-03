# Train the model and analyze the outputs.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
df = pd.read_csv('Single Layer Perceptron Dataset.csv')
df = df.dropna()  # Remove any rows with missing values

# Convert dataframe to numpy array
data = df[['Feature1', 'Feature2', 'Feature3', 'Class_Label']].values

X = data[:, :3]
y = data[:, 3].astype(int)
X_train, y_train = X[:8], y[:8]
X_val, y_val = X[8:], y[8:]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X, y, activation, epochs=1000, lr=0.1):
    np.random.seed(42)
    weights = np.random.rand(3)
    for epoch in range(epochs):
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            z = np.dot(xi, weights)
            a = activation(z)
            if activation == sigmoid:
                grad = (a - yi) * a * (1 - a) * xi
            elif activation == np.tanh:
                grad = (a - yi) * (1 - a**2) * xi
            else:
                grad = (a - yi) * (z > 0) * xi
            weights -= lr * grad
    return weights

weights_sigmoid = train(X_train, y_train, sigmoid)
weights_tanh = train(X_train, y_train, np.tanh)
weights_relu = train(X_train, y_train, lambda x: np.maximum(0, x))

def predict(X, weights, activation):
    preds = []
    for xi in X:
        z = np.dot(xi, weights)
        a = activation(z)
        if activation == sigmoid:
            pred = 1 if a >= 0.5 else 0
        elif activation == np.tanh:
            pred = 1 if a >= 0 else 0
        else:
            pred = 1 if a >= 0.5 else 0
        preds.append(pred)
    return np.array(preds)

print("Sigmoid Val:", np.mean(predict(X_val, weights_sigmoid, sigmoid) == y_val))
print("Tanh Val:", np.mean(predict(X_val, weights_tanh, np.tanh) == y_val))
print("ReLU Val:", np.mean(predict(X_val, weights_relu, lambda x: np.maximum(0, x)) == y_val))