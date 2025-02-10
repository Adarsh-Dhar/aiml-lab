import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

n_samples = 300


study_hours = np.random.normal(4, 1.5, n_samples)
study_hours = np.clip(study_hours, 0, 8)

# Generate previous scores between 0 and 100
previous_score = np.random.normal(70, 15, n_samples)
previous_score = np.clip(previous_score, 0, 100)

probability = 1 / (1 + np.exp(-(study_hours - 4) * 2 - (previous_score - 60) / 10))
pass_fail = (probability > 0.5).astype(int)
noise = np.random.random(n_samples) < 0.1
pass_fail[noise] = 1 - pass_fail[noise]

# Create DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'previous_score': previous_score,
    'passed': pass_fail
})

# Display dataset information
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

X = data[['study_hours', 'previous_score']]
y = data['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

plt.figure(figsize=(10, 8))
x_min, x_max = X['study_hours'].min() - 0.5, X['study_hours'].max() + 0.5
y_min, y_max = X['previous_score'].min() - 0.5, X['previous_score'].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X['study_hours'], X['previous_score'], c=y, alpha=0.8)
plt.xlabel('Study Hours')
plt.ylabel('Previous Score')
plt.title('Decision Boundary')
plt.show()

sample_student = pd.DataFrame({
    'study_hours': [6],
    'previous_score': [75]
})
prediction = model.predict(sample_student)
prob = model.predict_proba(sample_student)

print("\nSample Prediction:")
print(f"For a student who studied 6 hours and has a previous score of 75:")
print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Probability of passing: {prob[0][1]:.2%}")