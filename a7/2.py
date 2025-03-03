# Modify the Perceptron to Use Sigmoid Activation, Tanh Activation, ReLU Activation
# functions by replacing the Step function.

import numpy as np
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

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

activations = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}
results = {}

for name, activation in activations.items():
    np.random.seed(42)
    weights = np.random.rand(3)
    lr = 0.1
    epochs = 1000
    for epoch in range(epochs):
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]
            z = np.dot(xi, weights)
            a = activation(z)
            if name == 'sigmoid':
                grad = (a - yi) * a * (1 - a) * xi
            elif name == 'tanh':
                grad = (a - yi) * (1 - a**2) * xi
            elif name == 'relu':
                grad = (a - yi) * (z > 0) * xi
            weights -= lr * grad
    preds = []
    for xi in X_val:
        z = np.dot(xi, weights)
        a = activation(z)
        if name in ['sigmoid', 'relu']:
            pred = 1 if a >= 0.5 else 0
        elif name == 'tanh':
            pred = 1 if a >= 0 else 0
        preds.append(pred)
    acc = np.mean(np.array(preds) == y_val)
    results[name] = acc

print(results)