from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])
    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        return np.bincount(self.y_train[k_indices]).argmax()

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_values = [1, 50]
accuracies = []
for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.bar(['k=1 (Under-smoothed)', 'k=50 (Over-smoothed)'], accuracies, color=['purple', 'cyan'])
plt.ylabel('Accuracy')
plt.title('Effect of Extreme k Values')
plt.ylim(0.0, 1.0)
plt.show()