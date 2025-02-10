import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Iris.csv')
X = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy:", scores.mean())