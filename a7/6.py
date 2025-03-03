# Compare ReLU vs. Sigmoid for linear separability
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Read data from CSV
df = pd.read_csv('Single Layer Perceptron Dataset.csv')
df = df.dropna()  # Remove any rows with missing values

# Convert dataframe to numpy array
data = df[['Feature1', 'Feature2', 'Feature3', 'Class_Label']].values

X = data[:, 1:3]
y = data[:, 3].astype(int)

def train_relu(X, y, epochs=1000, lr=0.1):
    X_bias = np.c_[np.ones((len(X), 1)), X]
    weights = np.random.rand(3)
    for epoch in range(epochs):
        for i in range(len(X_bias)):
            xi = X_bias[i]
            yi = y[i]
            z = np.dot(xi, weights)
            a = np.maximum(0, z)
            grad = (a - yi) * (z > 0) * xi
            weights -= lr * grad
    return weights

weights_relu = train_relu(X, y)
weights_sigmoid = train_sigmoid(X, y)

Z_relu = np.array([1 if np.maximum(0, np.dot(xi, weights_relu)) >=0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
Z_sigmoid = np.array([1 if sigmoid(np.dot(xi, weights_sigmoid)) >=0.5 else 0 for xi in grid_bias]).reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_relu, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('ReLU')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_sigmoid, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Sigmoid')
plt.show()