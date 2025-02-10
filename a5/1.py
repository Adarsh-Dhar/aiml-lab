import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Iris.csv')
X = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy:", scores.mean())