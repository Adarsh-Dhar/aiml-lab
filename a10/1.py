import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the traffic dataset
df = pd.read_csv("traffic_dataset.csv")

# Handle categorical variables
categorical_columns = ['Weather_Conditions', 'Incidents_or_Events', 'Congestion_Level', 
                       'Optimal_Routing_Decisions', 'Traffic_Incidents', 'Peak_Hour_Prediction']

for col in categorical_columns:
    if col in df.columns:
        df[col] = pd.Categorical(df[col]).codes

# Drop non-numeric or redundant columns
columns_to_drop = ['Timestamp']
df_numeric = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Fill missing values if any
df_numeric = df_numeric.fillna(df_numeric.mean())

# Select key features that are most relevant for traffic analysis
# Using fewer dimensions can help DBSCAN perform better
key_features = ['Traffic_Volume', 'Traffic_Speed', 'Traffic_Density', 'Time_of_Day', 
                'Day_of_Week', 'Road_Length', 'Number_of_Lanes', 'Travel_Time']

# Check which features are actually in the dataframe
available_features = [col for col in key_features if col in df_numeric.columns]
print(f"Using features: {available_features}")

# Use only the selected features
features_df = df_numeric[available_features]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Step 1: Use PCA first to reduce dimensionality (helps DBSCAN work better)
pca = PCA(n_components=min(5, len(available_features)))
reduced_features = pca.fit_transform(scaled_features)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Step 2: Try wider range of eps values
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
min_samples_values = [3, 5, 10, 15, 20]

best_silhouette = -1
best_eps = None
best_min_samples = None
best_labels = None
best_n_clusters = 0

# Find optimal parameters using grid search
print("\nSearching for optimal parameters...")
dbscan_params_results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(reduced_features)
        
        # Calculate number of clusters (excluding noise points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only evaluate if we have actual clusters
        if n_clusters >= 2:
            try:
                # Filter out noise points for silhouette calculation
                if -1 in labels:
                    mask = labels != -1
                    score = silhouette_score(reduced_features[mask], labels[mask])
                else:
                    score = silhouette_score(reduced_features, labels)
                
                print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points, silhouette={score:.4f}")
                
                dbscan_params_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_points': n_noise,
                    'noise_percentage': n_noise / len(labels) * 100,
                    'silhouette': score
                })
                
                if score > best_silhouette:
                    best_silhouette = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
                    best_n_clusters = n_clusters
            except Exception as e:
                print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points - Error calculating silhouette")
        else:
            print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points - Skipping")

# Create DataFrame for results
dbscan_params_df = pd.DataFrame(dbscan_params_results)

# If we found valid parameters, use them
if best_eps is not None:
    print(f"\nBest parameters found: eps={best_eps}, min_samples={best_min_samples}")
    print(f"Number of clusters: {best_n_clusters}")
    print(f"Silhouette score: {best_silhouette:.4f}")
    
    # Use the best labels
    cluster_labels = best_labels
else:
    print("\nNo valid clustering parameters found. Using larger eps value...")
    
    # Try one more attempt with larger eps
    dbscan = DBSCAN(eps=5.0, min_samples=3)
    cluster_labels = dbscan.fit_predict(reduced_features)
    
    # Count number of samples in each cluster
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Using parameters: eps=5.0, min_samples=3")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters == 0:
        print("\nStill no clusters found. Let's try K-means instead for a fixed number of clusters.")
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_features)
        print(f"Using K-means with 5 clusters instead.")
        n_clusters = 5
        n_noise = 0

# Add cluster labels to original dataframe
df['Cluster'] = cluster_labels

# Visualization function similar to plot_parameter_search_results in @2.py
def plot_parameter_search_results(dbscan_params_df):
    """
    Plot the results of parameter search for DBSCAN.
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: DBSCAN eps vs silhouette score for different min_samples
    if not dbscan_params_df.empty:
        for min_samples, group in dbscan_params_df.groupby('min_samples'):
            axes[0, 0].plot(group['eps'], group['silhouette'], marker='o', 
                           linestyle='-', label=f'min_samples={min_samples}')
        
        axes[0, 0].set_xlabel('Epsilon (eps)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('DBSCAN: Epsilon vs Silhouette Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[0, 0].text(0.5, 0.5, 'No valid DBSCAN parameters found', 
                       horizontalalignment='center', verticalalignment='center')
    
    # Plot 2: DBSCAN number of clusters and noise percentage
    if not dbscan_params_df.empty:
        ax2 = axes[0, 1]
        best_params = dbscan_params_df.sort_values('silhouette', ascending=False).iloc[0]
        
        # Create a subset with only the best min_samples
        subset = dbscan_params_df[dbscan_params_df['min_samples'] == best_params['min_samples']]
        
        color = 'tab:blue'
        ax2.set_xlabel('Epsilon (eps)', color=color)
        ax2.set_ylabel('Number of Clusters', color=color)
        line1 = ax2.plot(subset['eps'], subset['n_clusters'], marker='o', 
                        color=color, label='Clusters')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax3 = ax2.twinx()
        color = 'tab:red'
        ax3.set_ylabel('Noise Percentage (%)', color=color)
        line2 = ax3.plot(subset['eps'], subset['noise_percentage'], marker='s', 
                        color=color, label='Noise %')
        ax3.tick_params(axis='y', labelcolor=color)
        
        ax2.set_title(f'DBSCAN: Clusters vs Noise (min_samples={best_params["min_samples"]})')
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid DBSCAN parameters found', 
                       horizontalalignment='center', verticalalignment='center')
    
    # Placeholder for K-means plots (since we don't have K-means results)
    axes[1, 0].text(0.5, 0.5, 'K-means results not available', 
                   horizontalalignment='center', verticalalignment='center')
    axes[1, 1].text(0.5, 0.5, 'K-means results not available', 
                   horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

# Visualize the clustering results with PCA (2D)
def plot_clusters_2d(data, labels, feature_names=None):
    """
    Visualize clusters in 2D using PCA.
    """
    # Apply PCA if data has more than 2 dimensions
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
    else:
        data_2d = data
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot clusters
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', 
                          marker='o', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.title('Traffic Data Clustering')
    plt.xlabel('PCA Component 1' if data.shape[1] > 2 else feature_names[0])
    plt.ylabel('PCA Component 2' if data.shape[1] > 2 else feature_names[1])
    legend = plt.legend(*scatter.legend_elements(), 
                        title="Clusters", loc="upper right")
    plt.add_artist(legend)
    plt.grid(True)
    plt.show()

# Plot parameter search results
plot_parameter_search_results(dbscan_params_df)

# Plot clusters
plot_clusters_2d(reduced_features, cluster_labels, feature_names=available_features)

# Analyze clusters in terms of traffic patterns
print("\nCluster Analysis:")
for i in sorted(set(cluster_labels)):
    if i == -1:
        print(f"\nNoise Points: {list(cluster_labels).count(-1)}")
    else:
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i} (Size: {len(cluster_data)})")
        
        # Display statistics for key metrics
        for feature in available_features:
            print(f"  Average {feature}: {cluster_data[feature].mean():.2f}")