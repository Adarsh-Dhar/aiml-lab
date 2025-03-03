# Compare decision boundaries of Sigmoid vs. Tanh.
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

def train_tanh(X, y, epochs=1000, lr=0.1):
    X_bias = np.c_[np.ones((len(X), 1)), X]
    weights = np.random.rand(3)
    for epoch in range(epochs):
        for i in range(len(X_bias)):
            xi = X_bias[i]
            yi = y[i]
            z = np.dot(xi, weights)
            a = np.tanh(z)
            grad = (a - yi) * (1 - a**2) * xi
            weights -= lr * grad
    return weights

weights_sigmoid = train_sigmoid(X, y)
weights_tanh = train_tanh(X, y)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_bias = np.c_[np.ones(grid.shape[0]), grid.reshape(-1, 2)]

Z_sigmoid = np.array([1 if sigmoid(np.dot(xi, weights_sigmoid)) >=0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
Z_tanh = np.array([1 if np.tanh(np.dot(xi, weights_tanh)) >=0 else 0 for xi in grid_bias]).reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_sigmoid, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Sigmoid')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_tanh, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Tanh')
plt.show()