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
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_bias = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]

activations = {
    'Step': weights_step,
    'Sigmoid': weights_sigmoid,
    'Tanh': weights_tanh,
    'ReLU': weights_relu
}

plt.figure(figsize=(15, 10))
for i, (name, w) in enumerate(activations.items(), 1):
    plt.subplot(2, 2, i)
    if name == 'Step':
        Z = np.array([step_activation(np.dot(xi, w)) for xi in grid_bias]).reshape(xx.shape)
    elif name == 'Sigmoid':
        Z = np.array([1 if sigmoid(np.dot(xi, w)) >=0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
    elif name == 'Tanh':
        Z = np.array([1 if np.tanh(np.dot(xi, w)) >=0 else 0 for xi in grid_bias]).reshape(xx.shape)
    elif name == 'ReLU':
        Z = np.array([1 if np.maximum(0, np.dot(xi, w)) >=0.5 else 0 for xi in grid_bias]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title(name)
plt.tight_layout()
plt.show()