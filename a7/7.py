# Plot the decision boundary for different activation functions
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

# Define activation functions
def step_activation(z):
    return 1 if z >= 0 else 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define training functions for each activation
def train_step(X, y, epochs=1000, lr=0.1):
    X_bias = np.c_[np.ones((len(X), 1)), X]
    weights = np.random.rand(3)
    for epoch in range(epochs):
        for i in range(len(X_bias)):
            xi = X_bias[i]
            yi = y[i]
            z = np.dot(xi, weights)
            a = 1 if z >= 0 else 0
            if a != yi:
                weights += lr * (yi - a) * xi
    return weights

def train_sigmoid(X, y, epochs=1000, lr=0.1):
    X_bias = np.c_[np.ones((len(X), 1)), X]
    weights = np.random.rand(3)
    for epoch in range(epochs):
        for i in range(len(X_bias)):
            xi = X_bias[i]
            yi = y[i]
            z = np.dot(xi, weights)
            a = sigmoid(z)
            grad = (a - yi) * a * (1 - a) * xi
            weights -= lr * grad
    return weights

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

# Train all models
weights_step = train_step(X, y)
weights_sigmoid = train_sigmoid(X, y)
weights_tanh = train_tanh(X, y)
weights_relu = train_relu(X, y)

# Prepare grid for visualization
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100), 
                    np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100))
grid_bias = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]

# Store the models in a dictionary
activations = {
    'Step': weights_step,
    'Sigmoid': weights_sigmoid,
    'Tanh': weights_tanh,
    'ReLU': weights_relu
}

# Plot decision boundaries
plt.figure(figsize=(15, 10))
for i, (name, w) in enumerate(activations.items(), 1):
    plt.subplot(2, 2, i)
    if name == 'Step':
        Z = np.array([1 if np.dot(xi, w) >= 0 else 0 for xi in grid_bias]).reshape(xx.shape)
    elif name == 'Sigmoid':
        Z = np.array([1 if sigmoid(np.dot(xi, w)) >= 0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
    elif name == 'Tanh':
        Z = np.array([1 if np.tanh(np.dot(xi, w)) >= 0 else 0 for xi in grid_bias]).reshape(xx.shape)
    elif name == 'ReLU':
        Z = np.array([1 if np.maximum(0, np.dot(xi, w)) >= 0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(name)
plt.tight_layout()
plt.show()