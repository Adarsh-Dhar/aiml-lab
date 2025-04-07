import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Generate a synthetic logistics dataset
# --------------------------------------------
def generate_logistics_dataset(n_samples=1000):
    """
    Generate a synthetic logistics dataset with features that might be found in logistics data.
    
    Features:
    - distance_km: Distance of delivery in kilometers
    - weight_kg: Weight of package in kilograms
    - delivery_time_min: Delivery time in minutes
    - package_value: Value of the package
    - customer_loyalty: Loyalty score of the customer
    """
    # Generate base features
    X_blobs, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], 
                           random_state=42, n_features=2)
    
    # Create DataFrame with interpretable features based on the blobs
    df = pd.DataFrame()
    
    # Convert blob features to logistics-like features
    df['distance_km'] = 5 + np.abs(X_blobs[:, 0] * 10)  # Scale to realistic distances (5-100km)
    df['weight_kg'] = 0.5 + np.abs(X_blobs[:, 1] * 5)   # Scale to realistic weights (0.5-20kg)
    
    # Add derived and additional features
    df['delivery_time_min'] = 15 + 2 * df['distance_km'] + df['weight_kg'] * 3 + np.random.normal(0, 10, n_samples)
    df['package_value'] = 10 + df['weight_kg'] * 15 + np.random.normal(0, 50, n_samples)
    df['customer_loyalty'] = np.random.uniform(1, 10, n_samples)
    
    # Add some non-linear patterns to make clustering more interesting
    mask = df['distance_km'] > 50
    df.loc[mask, 'delivery_time_min'] += 30  # Longer distances have disproportionately longer delivery times
    
    # Create structured segments in the data
    high_value_mask = df['package_value'] > 150
    df.loc[high_value_mask, 'customer_loyalty'] += 2  # High value packages tend to be from loyal customers
    
    return df

# Part 2: Data preprocessing
# --------------------------------------------
def preprocess_data(df):
    """
    Preprocess the logistics data for clustering.
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Check for missing values
    print("Missing values per column:")
    print(data.isnull().sum())
    
    # Handle outliers (using 3 standard deviations as a threshold)
    for column in data.columns:
        mean = data[column].mean()
        std = data[column].std()
        threshold = 3 * std
        outliers = data[column].apply(lambda x: abs(x - mean) > threshold)
        data.loc[outliers, column] = np.clip(data.loc[outliers, column], 
                                          mean - threshold, 
                                          mean + threshold)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

# Part 3: DBSCAN implementation
# --------------------------------------------
def apply_dbscan(data, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering to the data.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data)
    
    # Extract metrics
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    noise_points = list(dbscan_labels).count(-1)
    
    # Calculate silhouette score if there are at least 2 clusters and not all points are noise
    if n_clusters >= 2 and noise_points < len(data):
        # Remove noise points for silhouette calculation
        mask = dbscan_labels != -1
        if len(set(dbscan_labels[mask])) >= 2:  # At least 2 clusters excluding noise
            sil_score = silhouette_score(data[mask], dbscan_labels[mask])
        else:
            sil_score = None
    else:
        sil_score = None
    
    return dbscan_labels, n_clusters, noise_points, sil_score

# Part 4: K-means implementation
# --------------------------------------------
def apply_kmeans(data, n_clusters=4):
    """
    Apply K-means clustering to the data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(data)
    
    # Calculate silhouette score
    sil_score = silhouette_score(data, kmeans_labels)
    
    # Calculate inertia (within-cluster sum of squares)
    inertia = kmeans.inertia_
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    return kmeans_labels, sil_score, inertia, centers

# Part 5: Finding optimal parameters
# --------------------------------------------
def find_optimal_dbscan_params(data):
    """
    Find optimal eps and min_samples parameters for DBSCAN using a grid search.
    """
    best_silhouette = -1
    best_params = {}
    results = []
    
    eps_values = np.arange(0.3, 1.5, 0.1)
    min_samples_values = [5, 10, 15, 20, 25]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels, n_clusters, noise, sil_score = apply_dbscan(data, eps, min_samples)
            
            # Only consider parameter combinations that produce at least 2 clusters
            if n_clusters >= 2 and sil_score is not None:
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_points': noise,
                    'noise_percentage': noise / len(data) * 100,
                    'silhouette': sil_score
                })
                
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_params = {'eps': eps, 'min_samples': min_samples}
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print("Top DBSCAN parameter combinations by silhouette score:")
        print(results_df.sort_values('silhouette', ascending=False).head())
    else:
        print("No valid parameter combinations found. Consider expanding the search space.")
    
    return best_params

def find_optimal_k(data, max_k=10):
    """
    Find optimal number of clusters for K-means using the elbow method and silhouette score.
    """
    inertia_values = []
    silhouette_values = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        labels, sil_score, inertia, _ = apply_kmeans(data, k)
        inertia_values.append(inertia)
        silhouette_values.append(sil_score)
    
    # Create dataframe with results
    results_df = pd.DataFrame({
        'k': list(k_values),
        'inertia': inertia_values,
        'silhouette': silhouette_values
    })
    
    print("K-means results for different k values:")
    print(results_df)
    
    # Find optimal k using silhouette score
    optimal_k = k_values[np.argmax(silhouette_values)]
    
    return optimal_k, results_df

# Part 6: Visualization functions
# --------------------------------------------
def plot_clusters_2d(data, dbscan_labels, kmeans_labels, feature_names=None):
    """
    Visualize the clusters in 2D using PCA if needed.
    """
    # Apply PCA if data has more than 2 dimensions
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
    else:
        data_2d = data
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot DBSCAN results
    scatter1 = axes[0].scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis', 
                              marker='o', alpha=0.6, edgecolors='w', linewidths=0.5)
    axes[0].set_title('DBSCAN Clustering')
    axes[0].set_xlabel('PCA Component 1' if data.shape[1] > 2 else feature_names[0])
    axes[0].set_ylabel('PCA Component 2' if data.shape[1] > 2 else feature_names[1])
    legend1 = axes[0].legend(*scatter1.legend_elements(), 
                            title="Clusters", loc="upper right")
    axes[0].add_artist(legend1)
    
    # Plot K-means results
    scatter2 = axes[1].scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans_labels, cmap='plasma', 
                              marker='o', alpha=0.6, edgecolors='w', linewidths=0.5)
    axes[1].set_title('K-means Clustering')
    axes[1].set_xlabel('PCA Component 1' if data.shape[1] > 2 else feature_names[0])
    axes[1].set_ylabel('PCA Component 2' if data.shape[1] > 2 else feature_names[1])
    legend2 = axes[1].legend(*scatter2.legend_elements(), 
                            title="Clusters", loc="upper right")
    axes[1].add_artist(legend2)
    
    plt.tight_layout()
    plt.show()

def plot_parameter_search_results(dbscan_params_df, kmeans_results_df):
    """
    Plot the results of parameter search for both algorithms.
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
    
    # Plot 3: K-means elbow method
    axes[1, 0].plot(kmeans_results_df['k'], kmeans_results_df['inertia'], 
                   marker='o', linestyle='-', color='tab:green')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
    axes[1, 0].set_title('K-means: Elbow Method')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: K-means silhouette scores
    axes[1, 1].plot(kmeans_results_df['k'], kmeans_results_df['silhouette'], 
                   marker='o', linestyle='-', color='tab:purple')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Silhouette Score')
    axes[1, 1].set_title('K-means: Silhouette Scores')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def visualize_cluster_characteristics(df, dbscan_labels, kmeans_labels):
    """
    Visualize the characteristics of each cluster for better interpretation.
    """
    # Add cluster labels to original data
    df_with_clusters = df.copy()
    df_with_clusters['DBSCAN_Cluster'] = dbscan_labels
    df_with_clusters['KMeans_Cluster'] = kmeans_labels
    
    # Create a figure with multiple plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Function to create boxplots for each feature by cluster
    def create_boxplots(ax, data, cluster_col, feature_col, title):
        sns.boxplot(x=cluster_col, y=feature_col, data=data, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature_col)
    
    # Features to visualize
    features = df.columns
    
    # Only keep non-noise points for DBSCAN visualization
    dbscan_data = df_with_clusters[df_with_clusters['DBSCAN_Cluster'] != -1].copy()
    
    # If there are valid DBSCAN clusters
    if len(dbscan_data) > 0 and len(set(dbscan_data['DBSCAN_Cluster'])) > 1:
        # Plot 1: BoxPlot of distance by DBSCAN cluster
        create_boxplots(axes[0, 0], dbscan_data, 'DBSCAN_Cluster', 
                       features[0], f'Distribution of {features[0]} by DBSCAN Cluster')
        
        # Plot 2: BoxPlot of weight by DBSCAN cluster
        create_boxplots(axes[0, 1], dbscan_data, 'DBSCAN_Cluster', 
                       features[1], f'Distribution of {features[1]} by DBSCAN Cluster')
        
        # Plot 3: Scatter plot of two main features colored by DBSCAN cluster
        scatter = axes[0, 2].scatter(dbscan_data[features[0]], dbscan_data[features[1]], 
                                    c=dbscan_data['DBSCAN_Cluster'], cmap='viridis', 
                                    alpha=0.6, edgecolors='w', linewidths=0.5)
        axes[0, 2].set_title(f'{features[0]} vs {features[1]} by DBSCAN Cluster')
        axes[0, 2].set_xlabel(features[0])
        axes[0, 2].set_ylabel(features[1])
        legend = axes[0, 2].legend(*scatter.legend_elements(), 
                                  title="Clusters", loc="upper right")
        axes[0, 2].add_artist(legend)
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'No valid DBSCAN clusters', 
                           horizontalalignment='center', verticalalignment='center')
    
    # Plot 4: BoxPlot of distance by K-means cluster
    create_boxplots(axes[1, 0], df_with_clusters, 'KMeans_Cluster', 
                   features[0], f'Distribution of {features[0]} by K-means Cluster')
    
    # Plot 5: BoxPlot of weight by K-means cluster
    create_boxplots(axes[1, 1], df_with_clusters, 'KMeans_Cluster', 
                   features[1], f'Distribution of {features[1]} by K-means Cluster')
    
    # Plot 6: Scatter plot of two main features colored by K-means cluster
    scatter = axes[1, 2].scatter(df_with_clusters[features[0]], df_with_clusters[features[1]], 
                                c=df_with_clusters['KMeans_Cluster'], cmap='plasma', 
                                alpha=0.6, edgecolors='w', linewidths=0.5)
    axes[1, 2].set_title(f'{features[0]} vs {features[1]} by K-means Cluster')
    axes[1, 2].set_xlabel(features[0])
    axes[1, 2].set_ylabel(features[1])
    legend = axes[1, 2].legend(*scatter.legend_elements(), 
                              title="Clusters", loc="upper right")
    axes[1, 2].add_artist(legend)
    
    plt.tight_layout()
    plt.show()

# Part 7: Compare clustering results
# --------------------------------------------
def compare_clustering_algorithms(df, dbscan_labels, kmeans_labels):
    """
    Analyze and compare the clustering results from DBSCAN and K-means.
    """
    print("\n" + "="*50)
    print("COMPARISON OF CLUSTERING ALGORITHMS")
    print("="*50)
    
    # Calculate number of clusters
    dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    kmeans_clusters = len(set(kmeans_labels))
    
    print(f"DBSCAN found {dbscan_clusters} clusters")
    print(f"K-means was set to find {kmeans_clusters} clusters")
    
    # Calculate number of noise points in DBSCAN
    noise_points = list(dbscan_labels).count(-1)
    noise_percentage = noise_points / len(dbscan_labels) * 100
    print(f"DBSCAN identified {noise_points} noise points ({noise_percentage:.2f}% of data)")
    
    # Create cross-tabulation to see how points are distributed between the two methods
    df_with_clusters = pd.DataFrame({
        'DBSCAN': ['Noise' if label == -1 else f'Cluster {label}' for label in dbscan_labels],
        'KMeans': [f'Cluster {label}' for label in kmeans_labels]
    })
    
    crosstab = pd.crosstab(df_with_clusters['DBSCAN'], 
                          df_with_clusters['KMeans'], 
                          normalize='index') * 100
    
    print("\nCross-tabulation of cluster assignments (% of DBSCAN cluster in each K-means cluster):")
    print(crosstab)
    
    # Add the clustering results to the original dataframe and get cluster statistics
    df_with_clusters = df.copy()
    df_with_clusters['DBSCAN_Cluster'] = dbscan_labels
    df_with_clusters['KMeans_Cluster'] = kmeans_labels
    
    print("\nDBSCAN Cluster Statistics:")
    dbscan_stats = df_with_clusters.groupby('DBSCAN_Cluster').mean()
    print(dbscan_stats)
    
    print("\nK-means Cluster Statistics:")
    kmeans_stats = df_with_clusters.groupby('KMeans_Cluster').mean()
    print(kmeans_stats)
    
    # Calculate cluster sizes
    dbscan_sizes = df_with_clusters['DBSCAN_Cluster'].value_counts().sort_index()
    kmeans_sizes = df_with_clusters['KMeans_Cluster'].value_counts().sort_index()
    
    print("\nDBSCAN Cluster Sizes:")
    for cluster, size in dbscan_sizes.items():
        cluster_name = 'Noise' if cluster == -1 else f'Cluster {cluster}'
        print(f"{cluster_name}: {size} points ({size/len(df_with_clusters)*100:.2f}%)")
    
    print("\nK-means Cluster Sizes:")
    for cluster, size in kmeans_sizes.items():
        print(f"Cluster {cluster}: {size} points ({size/len(df_with_clusters)*100:.2f}%)")
    
    # Qualitative comparison
    print("\nQualitative Comparison:")
    print("-"*50)
    print("DBSCAN Advantages:")
    print("- Does not require specifying the number of clusters in advance")
    print("- Can find arbitrarily shaped clusters")
    print("- Robust to outliers (marks them as noise)")
    print("- Works well when clusters have different densities")
    
    print("\nDBSCAN Disadvantages:")
    print("- Sensitive to parameter selection (eps and min_samples)")
    print("- May struggle with datasets with varying densities")
    print("- Difficulty handling high-dimensional data")
    
    print("\nK-means Advantages:")
    print("- Simple to implement and understand")
    print("- Scales well to large datasets")
    print("- Generally faster than DBSCAN")
    print("- Works well with globular clusters")
    
    print("\nK-means Disadvantages:")
    print("- Requires specifying number of clusters in advance")
    print("- Assumes clusters are spherical and of similar size")
    print("- Sensitive to outliers")
    print("- May converge to local optima")
    
    return dbscan_stats, kmeans_stats, crosstab

# Part 8: Main function
# --------------------------------------------
def main():
    """
    Main function to run the entire analysis.
    """
    print("="*50)
    print("CLUSTERING ANALYSIS: DBSCAN vs K-MEANS")
    print("="*50)
    
    # Step 1: Generate or load logistics data
    print("\nGenerating synthetic logistics dataset...\n")
    df = generate_logistics_dataset(n_samples=1000)
    print("Dataset summary:")
    print(df.describe())
    
    # Step 2: Preprocess the data
    print("\nPreprocessing data...\n")
    scaled_data, scaler = preprocess_data(df)
    
    # Step 3: Find optimal parameters for both algorithms
    print("\nFinding optimal DBSCAN parameters...\n")
    dbscan_params_results = []
    eps_values = np.arange(0.3, 1.5, 0.1)
    min_samples_values = [5, 10, 15, 20, 25]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels, n_clusters, noise, sil_score = apply_dbscan(scaled_data, eps, min_samples)
            
            dbscan_params_results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_points': noise,
                'noise_percentage': noise / len(scaled_data) * 100,
                'silhouette': sil_score if sil_score is not None else np.nan
            })
    
    dbscan_params_df = pd.DataFrame(dbscan_params_results)
    
    # Filter out rows with NaN silhouette scores
    valid_params_df = dbscan_params_df.dropna(subset=['silhouette'])
    
    if not valid_params_df.empty:
        best_params = valid_params_df.loc[valid_params_df['silhouette'].idxmax()]
        best_eps = best_params['eps']
        best_min_samples = int(best_params['min_samples'])
        print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
        print(f"Number of clusters: {int(best_params['n_clusters'])}")
        print(f"Silhouette score: {best_params['silhouette']:.4f}")
    else:
        print("No valid DBSCAN parameters found. Using default values.")
        best_eps = 0.5
        best_min_samples = 5
    
    print("\nFinding optimal K for K-means...\n")
    optimal_k, kmeans_results_df = find_optimal_k(scaled_data)
    print(f"Optimal number of clusters for K-means: {optimal_k}")
    
    # Step 4: Apply both clustering algorithms with optimal parameters
    print("\nApplying DBSCAN with optimal parameters...\n")
    dbscan_labels, dbscan_n_clusters, dbscan_noise, dbscan_sil = apply_dbscan(
        scaled_data, eps=best_eps, min_samples=best_min_samples)
    
    print("\nApplying K-means with optimal K...\n")
    kmeans_labels, kmeans_sil, kmeans_inertia, kmeans_centers = apply_kmeans(
        scaled_data, n_clusters=optimal_k)
    
    # Step 5: Compare results
    dbscan_stats, kmeans_stats, crosstab = compare_clustering_algorithms(
        df, dbscan_labels, kmeans_labels)
    
    # Step 6: Visualize results
    print("\nVisualizing clustering results...\n")
    # Plotting clusters in 2D
    plot_clusters_2d(scaled_data, dbscan_labels, kmeans_labels, feature_names=df.columns)
    
    # Plotting parameter search results
    plot_parameter_search_results(valid_params_df, kmeans_results_df)
    
    # Visualizing cluster characteristics
    visualize_cluster_characteristics(df, dbscan_labels, kmeans_labels)
    
    # Step 7: Return results
    results = {
        'original_data': df,
        'scaled_data': scaled_data,
        'dbscan_labels': dbscan_labels,
        'kmeans_labels': kmeans_labels,
        'dbscan_params': {'eps': best_eps, 'min_samples': best_min_samples},
        'kmeans_params': {'k': optimal_k},
        'dbscan_silhouette': dbscan_sil,
        'kmeans_silhouette': kmeans_sil
    }
    
    return results

# Run the analysis
if __name__ == "__main__":
    results = main()