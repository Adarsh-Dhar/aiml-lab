import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read and explore the data
df = pd.read_csv('Fish.csv')

# Display initial information about the dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# If 'Species' is a categorical column, we should convert it to numeric
# If it's relevant for prediction, you could encode it, but for correlation matrix calculation,
# it needs to be excluded or encoded.
df_numeric = df.select_dtypes(include=[np.number])

# Create correlation matrix visualization (only numeric columns)
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Fish Measurements')
plt.show()

# Distribution of weights by species (handle categorical 'Species' column)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Species', y='Weight', data=df)
plt.xticks(rotation=45)
plt.title('Weight Distribution by Fish Species')
plt.show()

# Prepare features and target
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Feature', y='Coefficient')
plt.title('Feature Importance (Coefficients)')
plt.xticks(rotation=45)
plt.show()

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')
plt.title('Actual vs Predicted Fish Weight')
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Weight')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot')
plt.show()

# Example predictions
print("\nSample Predictions:")
sample_fish = pd.DataFrame({
    'Length1': [23.2, 25.4, 26.3],
    'Length2': [25.4, 27.5, 29.0],
    'Length3': [30.0, 28.9, 33.5],
    'Height': [11.52, 8.323, 12.73],
    'Width': [4.02, 5.1373, 4.4555]
})

predictions = model.predict(sample_fish)
for i, pred in enumerate(predictions):
    print(f"\nFish {i+1}:")
    print(f"Measurements:")
    for col in sample_fish.columns:
        print(f"{col}: {sample_fish.iloc[i][col]}")
    print(f"Predicted Weight: {pred:.2f}g")
