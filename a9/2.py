import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load customer segmentation dataset (using a sample dataset)
data = pd.read_csv('Mall_Customers.csv')

# Select relevant features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using silhouette score
max_clusters = 10
silhouette_scores = []

for n_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the optimal number of clusters (highest silhouette score)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Apply K-Means with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
                      c=data['Cluster'], cmap='viridis')
plt.title(f'Customer Segmentation (K-Means, {optimal_clusters} Clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(scatter, label='Cluster')

# Cluster analysis
cluster_summary = data.groupby('Cluster')[features].mean()
print("Cluster Summary:\n", cluster_summary)

plt.tight_layout()
plt.show()