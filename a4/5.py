import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Create the user dataset
data = {
    'User_ID': range(1, 501),
    'Gender': np.random.choice(['Male', 'Female'], size=500),
    'Age': np.random.normal(35, 10, 500),  # Age centered around 35
    'EstimatedSalary': np.random.normal(60000, 20000, 500)  # Salary centered around 60k
}

# Create DataFrame
df = pd.DataFrame(data)

# Clean up age data to be realistic
df['Age'] = np.clip(df['Age'], 18, 70).astype(int)

# Create purchase decision based on logical rules
# Higher probability of purchase for higher salary and middle-aged customers
purchase_probability = (
    0.3 +  # Base probability
    0.4 * (df['EstimatedSalary'] > 60000) +  # Salary factor
    0.3 * ((df['Age'] >= 25) & (df['Age'] <= 50)) +  # Age factor
    0.2 * (df['Gender'] == 'Male')  # Gender factor
)

df['Purchased'] = (np.random.random(500) < purchase_probability).astype(int)

# Display initial data exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Convert Gender to numeric (0 for Female, 1 for Male)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Prepare features and target
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Model evaluation
print("\nModel Performance:")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
importance = pd.DataFrame({
    'Feature': ['Gender', 'Age', 'Estimated Salary'],
    'Coefficient': abs(model.coef_[0])
})
plt.figure(figsize=(8, 6))
sns.barplot(data=importance, x='Feature', y='Coefficient')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

# Age vs Salary plot with purchase outcome
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Age'], df['EstimatedSalary'], 
                     c=df['Purchased'], cmap='coolwarm', 
                     alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Age vs Salary with Purchase Decision')
plt.colorbar(scatter)
plt.show()

# Purchase distribution by gender
plt.figure(figsize=(8, 6))
purchase_by_gender = pd.crosstab(df['Gender'], df['Purchased'])
purchase_by_gender.plot(kind='bar', stacked=True)
plt.title('Purchase Distribution by Gender')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Count')
plt.legend(['No Purchase', 'Purchased'])
plt.show()

# Example predictions
print("\nSample Predictions:")
sample_customers = pd.DataFrame({
    'Gender': [1, 0, 1],  # Male, Female, Male
    'Age': [30, 45, 25],
    'EstimatedSalary': [70000, 80000, 45000]
})

# Scale the sample data
sample_customers_scaled = scaler.transform(sample_customers)

# Make predictions
sample_predictions = model.predict(sample_customers_scaled)
sample_probabilities = model.predict_proba(sample_customers_scaled)

print("\nSample Customer Predictions:")
for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities)):
    gender = "Male" if sample_customers.iloc[i]['Gender'] == 1 else "Female"
    print(f"\nCustomer {i+1}:")
    print(f"Gender: {gender}")
    print(f"Age: {sample_customers.iloc[i]['Age']}")
    print(f"Salary: ${sample_customers.iloc[i]['EstimatedSalary']:,.2f}")
    print(f"Prediction: {'Will Purchase' if pred == 1 else 'Will Not Purchase'}")
    print(f"Probability of Purchase: {prob[1]:.2%}")