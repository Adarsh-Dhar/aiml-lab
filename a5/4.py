import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Iris.csv')
X = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

model = RandomForestClassifier()
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=skf)
print("Accuracy:", scores.mean())