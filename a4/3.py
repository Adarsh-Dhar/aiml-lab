import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

data = {
    'ID': range(1, 501),
    'flat': np.random.uniform(50000, 200000, 500),  
    'houses': np.random.uniform(100000, 500000, 500)  
}


df = pd.DataFrame(data)


df['purchases'] = (0.4 * df['flat'] + 
                  0.6 * df['houses'] + 
                  np.random.normal(0, 10000, 500))  


print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())


X = df[['flat', 'houses']]
y = df['purchases']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("\nTraining set features shape:", X_train.shape)
print("Training set target shape:", y_train.shape)
print("Testing set features shape:", X_test.shape)
print("Testing set target shape:", y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Purchases')
plt.ylabel('Predicted Purchases')
plt.title('Actual vs Predicted Purchase Values')
plt.show()


plt.figure(figsize=(8, 6))
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model.coef_)
})
sns.barplot(data=importance, x='Feature', y='Coefficient')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df[['flat', 'houses', 'purchases']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


sample_input = pd.DataFrame({
    'flat': [100000],
    'houses': [300000]
})
sample_prediction = model.predict(sample_input)
print("\nSample Prediction:")
print(f"For a flat worth $100,000 and a house worth $300,000")
print(f"Predicted purchase value: ${sample_prediction[0]:,.2f}")