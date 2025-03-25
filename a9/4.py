from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])
    def _predict(self, x):
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        k_indices = np.argsort(distances)[:self.k]
        return np.bincount(self.y_train[k_indices]).argmax()

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_euc = KNN(k=5, metric='euclidean')
knn_man = KNN(k=5, metric='manhattan')
knn_euc.fit(X_train, y_train)
knn_man.fit(X_train, y_train)

acc_euc = accuracy_score(y_test, knn_euc.predict(X_test))
acc_man = accuracy_score(y_test, knn_man.predict(X_test))

plt.figure(figsize=(8, 5))
plt.bar(['Euclidean', 'Manhattan'], [acc_euc, acc_man], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Distance Metric Comparison')
plt.ylim(0.8, 1.0)
plt.show()