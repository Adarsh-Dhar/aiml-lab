# Implement Single layer perceptron for the given data set. A total of 13 samples with three
# features and one class label. The class label is defined in binary 0 and 1. The training
# dataset contains eight data samples, while the validation dataset contains five.
 
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

np.random.seed(42)
weights = np.random.rand(3)
lr = 0.1
epochs = 100

for epoch in range(epochs):
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]
        pred = np.dot(xi, weights)
        pred_step = 1 if pred >= 0 else 0
        if pred_step != yi:
            weights += lr * (yi - pred_step) * xi

correct = 0
for i in range(len(X_val)):
    xi = X_val[i]
    yi = y_val[i]
    pred = np.dot(xi, weights)
    pred_step = 1 if pred >= 0 else 0
    if pred_step == yi:
        correct += 1
print(f"Validation Accuracy: {correct / len(X_val)}")