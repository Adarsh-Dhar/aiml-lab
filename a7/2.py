# Modify the Perceptron to Use Sigmoid Activation, Tanh Activation, ReLU Activation
# functions by replacing the Step function.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read data from CSV
df = pd.read_csv('Single Layer Perceptron Dataset.csv')
df = df.dropna()  # Remove any rows with missing values

# Convert dataframe to numpy array
data = df[['Feature1', 'Feature2', 'Feature3', 'Class_Label']].values

# Feature scaling for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[:, :3])

X = X_scaled
y = data[:, 3].astype(int)
X_train, y_train = X[:8], y[:8]
X_val, y_val = X[8:], y[8:]

# Define activation functions
def step(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Define activation function derivatives
def sigmoid_derivative(a):
    return a * (1 - a)

def tanh_derivative(a):
    return 1 - a**2

def relu_derivative(z):
    return 1 if z > 0 else 0

# Group activation functions and their derivatives
activations = {
    'step': (step, None),  # No derivative for step function
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative)
}

results = {}
all_training_history = {}
all_weights = {}

# For each activation function
for name, (activation, derivative) in activations.items():
    np.random.seed(42)
    weights = np.random.rand(3)
    bias = np.random.rand()
    lr = 0.01
    epochs = 1000
    
    train_accuracy_history = []
    val_accuracy_history = []
    
    print(f"\nTraining with {name} activation:")
    
    for epoch in range(epochs):
        # Training phase
        correct_train = 0
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]
            z = np.dot(xi, weights) + bias
            
            if name == 'step':
                # Step function (original perceptron)
                a = 1 if z >= 0 else 0
                if a != yi:
                    weights += lr * (yi - a) * xi
                    bias += lr * (yi - a)
            else:
                # Other activation functions (gradient-based)
                a = activation(z)
                
                # Determine the error and update
                if name == 'sigmoid':
                    error = yi - a
                    weights += lr * error * sigmoid_derivative(a) * xi
                    bias += lr * error * sigmoid_derivative(a)
                elif name == 'tanh':
                    error = yi - ((a + 1) / 2)  # Scale tanh output from [-1,1] to [0,1]
                    weights += lr * error * tanh_derivative(a) * xi
                    bias += lr * error * tanh_derivative(a)
                elif name == 'relu':
                    # For ReLU, we need to manually implement prediction threshold
                    pred = 1 if a >= 0.5 else 0
                    error = yi - pred
                    weights += lr * error * (1 if z > 0 else 0) * xi
                    bias += lr * error * (1 if z > 0 else 0)
            
            # Compute prediction for training accuracy
            if name == 'step':
                pred = 1 if z >= 0 else 0
            elif name == 'sigmoid':
                pred = 1 if sigmoid(z) >= 0.5 else 0
            elif name == 'tanh':
                pred = 1 if tanh(z) >= 0 else 0
            elif name == 'relu':
                pred = 1 if relu(z) >= 0.5 else 0
                
            if pred == yi:
                correct_train += 1
        
        # Validation phase
        correct_val = 0
        for i in range(len(X_val)):
            xi = X_val[i]
            yi = y_val[i]
            z = np.dot(xi, weights) + bias
            
            if name == 'step':
                pred = 1 if z >= 0 else 0
            elif name == 'sigmoid':
                pred = 1 if sigmoid(z) >= 0.5 else 0
            elif name == 'tanh':
                pred = 1 if tanh(z) >= 0 else 0
            elif name == 'relu':
                pred = 1 if relu(z) >= 0.5 else 0
                
            if pred == yi:
                correct_val += 1
        
        train_acc = correct_train / len(X_train)
        val_acc = correct_val / len(X_val)
        
        train_accuracy_history.append(train_acc)
        val_accuracy_history.append(val_acc)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Final evaluation
    preds = []
    for xi in X_val:
        z = np.dot(xi, weights) + bias
        if name == 'step':
            pred = 1 if z >= 0 else 0
        elif name == 'sigmoid':
            pred = 1 if sigmoid(z) >= 0.5 else 0
        elif name == 'tanh':
            pred = 1 if tanh(z) >= 0 else 0
        elif name == 'relu':
            pred = 1 if relu(z) >= 0.5 else 0
        preds.append(pred)
    
    final_acc = np.mean(np.array(preds) == y_val)
    results[name] = final_acc
    all_training_history[name] = (train_accuracy_history, val_accuracy_history)
    all_weights[name] = (weights, bias)
    
    print(f"Final {name} validation accuracy: {final_acc:.4f}")

# Print overall results
print("\nFinal Results:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Visualize training progress for all activation functions
plt.figure(figsize=(15, 10))

for i, (name, (train_history, val_history)) in enumerate(all_training_history.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(range(epochs), train_history, label='Training Accuracy')
    plt.plot(range(epochs), val_history, label='Validation Accuracy')
    plt.title(f'{name.capitalize()} Activation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot decision boundaries (2D visualization using first 2 features)
if X.shape[1] >= 2:
    plt.figure(figsize=(15, 10))
    
    for i, (name, (weights, bias)) in enumerate(all_weights.items(), 1):
        plt.subplot(2, 2, i)
        
        # Plot training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', 
                    marker='o', s=80, edgecolors='k')
        
        # Plot validation data
        plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='coolwarm', 
                    marker='x', s=80)
        
        # Plot decision boundary (approximate for 2D visualization)
        if abs(weights[1]) > 1e-10:  # Avoid division by zero
            xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
            # Decision boundary: w0*x0 + w1*x1 + bias = 0
            # Solve for x1: x1 = (-w0*x0 - bias) / w1
            yy = (-weights[0] * xx - bias) / weights[1]
            plt.plot(xx, yy, 'k-', label='Decision Boundary')
        
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.title(f'{name.capitalize()} Activation')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()