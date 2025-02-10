import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Iris.csv')
X = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

model = LogisticRegression(max_iter=200)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf,scoring= "accuracy")
print("Accuracy:", scores.mean())