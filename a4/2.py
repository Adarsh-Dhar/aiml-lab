import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(100)

# Create the sales dataset
data = {
    'ID': range(1, 501),
    'TV': np.random.uniform(10, 300, 500),
    'Radio': np.random.uniform(5, 50, 500),
    'Newspaper': np.random.uniform(0, 100, 500)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate Sales with realistic relationships
df['Sales'] = (0.15 * df['TV'] + 
               0.30 * df['Radio'] + 
               0.05 * df['Newspaper'] + 
               np.random.normal(0, 10, 500))

# Display first few rows and dataset info
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Identify features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of training and testing sets
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# Feature importance visualization
plt.figure(figsize=(8, 6))
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model.coef_)
})
sns.barplot(data=importance, x='Feature', y='Coefficient')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.show()