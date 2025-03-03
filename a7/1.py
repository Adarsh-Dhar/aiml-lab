# Implement Single layer perceptron for the given data set. A total of 13 samples with three
# features and one class label. The class label is defined in binary 0 and 1. The training
# dataset contains eight data samples, while the validation dataset contains five.
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
df = pd.read_csv('Single Layer Perceptron Dataset.csv')
df = df.dropna()  # Remove any rows with missing values

# Convert dataframe to numpy array
data = df[['Feature1', 'Feature2', 'Feature3', 'Class_Label']].values

# Feature scaling for better convergence
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[:, :3])

X = X_scaled
y = data[:, 3].astype(int)

# Split into training (8 samples) and validation (5 samples) sets
X_train, y_train = X[:8], y[:8]
X_val, y_val = X[8:], y[8:]

# Initialize weights and bias
np.random.seed(42)
weights = np.random.rand(3)
bias = np.random.rand()
lr = 0.1
epochs = 100

# Lists to track performance
train_accuracy_history = []
val_accuracy_history = []
epoch_history = []

# Training loop
for epoch in range(epochs):
    # Training
    train_correct = 0
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]
        
        # Forward pass
        activation = np.dot(xi, weights) + bias
        prediction = 1 if activation >= 0 else 0
        
        # Update weights only if prediction is wrong
        if prediction != yi:
            weights += lr * (yi - prediction) * xi
            bias += lr * (yi - prediction)
        
        # Track training accuracy
        if prediction == yi:
            train_correct += 1
    
    # Validation
    val_correct = 0
    for i in range(len(X_val)):
        xi = X_val[i]
        yi = y_val[i]
        
        # Forward pass
        activation = np.dot(xi, weights) + bias
        prediction = 1 if activation >= 0 else 0
        
        # Track validation accuracy
        if prediction == yi:
            val_correct += 1
    
    # Calculate and store accuracies
    train_accuracy = train_correct / len(X_train)
    val_accuracy = val_correct / len(X_val)
    
    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)
    epoch_history.append(epoch)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Final evaluation
final_train_predictions = []
for i in range(len(X_train)):
    xi = X_train[i]
    activation = np.dot(xi, weights) + bias
    pred = 1 if activation >= 0 else 0
    final_train_predictions.append(pred)

final_val_predictions = []
for i in range(len(X_val)):
    xi = X_val[i]
    activation = np.dot(xi, weights) + bias
    pred = 1 if activation >= 0 else 0
    final_val_predictions.append(pred)

# Calculate final accuracy
final_train_accuracy = sum(p == a for p, a in zip(final_train_predictions, y_train)) / len(y_train)
final_val_accuracy = sum(p == a for p, a in zip(final_val_predictions, y_val)) / len(y_val)

print("\nFinal Results:")
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")

# Visualize training progress
plt.figure(figsize=(10, 6))
plt.plot(epoch_history, train_accuracy_history, label='Training Accuracy')
plt.plot(epoch_history, val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Single Layer Perceptron Training Progress')
plt.legend()
plt.grid(True)
plt.show()

# Print confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Training confusion matrix
train_cm = confusion_matrix(y_train, final_train_predictions)
val_cm = confusion_matrix(y_val, final_val_predictions)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Training Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Display individual data points with decision boundary (using first two features)
if X.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', 
                marker='o', s=100, edgecolors='k', label='Training data')
    
    # Plot validation data
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='coolwarm', 
                marker='x', s=100, label='Validation data')
    
    # Plot decision boundary (approximate for 2D visualization)
    if abs(weights[1]) > 1e-10:  # Avoid division by zero
        xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        # Decision boundary: w0*x0 + w1*x1 + bias = 0
        # Solve for x1: x1 = (-w0*x0 - bias) / w1
        yy = (-weights[0] * xx - bias) / weights[1]
        plt.plot(xx, yy, 'k-', label='Decision Boundary')
    
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Single Layer Perceptron Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()