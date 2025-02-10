import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("landslide_data.csv")

print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
df.info()

relevant_features = [
    'landslide_trigger',
    'landslide_size',
    'landslide_setting',
    'fatality_count',
    'injury_count',
    'latitude',
    'longitude',
    'admin_division_population'
]

target_column = 'landslide_category'

df_selected = df[relevant_features + [target_column]].copy()

print("\nMissing Values in Selected Features:")
print(df_selected.isnull().sum())

numeric_columns = df_selected.select_dtypes(include=['float64', 'int64']).columns
df_selected[numeric_columns] = df_selected[numeric_columns].fillna(df_selected[numeric_columns].mean())

categorical_columns = df_selected.select_dtypes(include=['object']).columns
print("\nCategorical columns selected:", categorical_columns.tolist())

label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df_selected[col].fillna('unknown', inplace=True)
    df_selected[col] = label_encoders[col].fit_transform(df_selected[col])

X = df_selected.drop(columns=[target_column])
y = df_selected[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
feature_importance = np.abs(model.theta_[1] - model.theta_[0])
feature_importance = pd.Series(feature_importance, index=X.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

def predict_new_location():
    print("\nEnter values for new prediction:")
    new_data = {}
    
    for col in relevant_features:
        if col in categorical_columns:
            print(f"\nAvailable categories for {col}:")
            print(label_encoders[col].classes_.tolist())
            val = input(f"Enter value for {col}: ")
            if val in label_encoders[col].classes_:
                new_data[col] = [label_encoders[col].transform([val])[0]]
            else:
                print(f"Invalid category. Using 'unknown' for {col}")
                new_data[col] = [label_encoders[col].transform(['unknown'])[0]]
        else:
            try:
                val = float(input(f"Enter value for {col}: "))
                new_data[col] = [val]
            except ValueError:
                print(f"Invalid input. Using mean value for {col}")
                new_data[col] = [df_selected[col].mean()]
    
    new_location = pd.DataFrame(new_data)
    new_location_scaled = scaler.transform(new_location)
    
    probability = model.predict_proba(new_location_scaled)
    prediction = model.predict(new_location_scaled)
    
    print("\nPrediction Results:")
    for i, category in enumerate(label_encoders[target_column].classes_):
        print(f"Probability of {category}: {probability[0][i]:.2f}")
    
    print(f"\nPredicted Category: {label_encoders[target_column].inverse_transform(prediction)[0]}")

predict_new_location()

plt.figure(figsize=(10, 6))
feature_correlations = df_selected.corr()
sns.heatmap(feature_correlations, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.show()