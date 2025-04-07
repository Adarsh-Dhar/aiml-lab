import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic customer data
def generate_customer_data(n_samples=1000):
    """Generate synthetic customer data for segmentation analysis."""
    # Customer annual income (in thousands)
    income = np.concatenate([
        np.random.normal(45, 15, int(n_samples*0.5)),  # Lower-middle income
        np.random.normal(100, 25, int(n_samples*0.3)),  # Higher income
        np.random.normal(150, 20, int(n_samples*0.2))   # Very high income
    ])
    
    # Annual purchase amount (in hundreds)
    purchase = np.concatenate([
        np.random.normal(20, 10, int(n_samples*0.5)),   # Low spenders
        np.random.normal(80, 20, int(n_samples*0.3)),   # Medium spenders
        np.random.normal(200, 50, int(n_samples*0.2))   # High spenders
    ])
    
    # Website visits per month
    web_visits = np.concatenate([
        np.random.poisson(2, int(n_samples*0.5)),       # Rare visitors
        np.random.poisson(8, int(n_samples*0.3)),       # Regular visitors
        np.random.poisson(20, int(n_samples*0.2))       # Frequent visitors
    ])
    
    # Store visits per month
    store_visits = np.concatenate([
        np.random.poisson(0.5, int(n_samples*0.5)),     # Rare store visitors
        np.random.poisson(3, int(n_samples*0.3)),       # Occasional store visitors
        np.random.poisson(10, int(n_samples*0.2))       # Frequent store visitors
    ])
    
    # Create dataframe
    df = pd.DataFrame({
        'Income': income,
        'PurchaseAmount': purchase,
        'WebVisits': web_visits,
        'StoreVisits': store_visits
    })
    
    # Add some noise and outliers
    outliers_idx = np.random.choice(range(len(df)), size=int(0.03*len(df)), replace=False)
    df.loc[outliers_idx, 'Income'] = np.random.uniform(200, 500, size=len(outliers_idx))
    df.loc[outliers_idx, 'PurchaseAmount'] = np.random.uniform(300, 800, size=len(outliers_idx))
    
    return df

# Generate data
customer_data = generate_customer_data(1000)
print("Dataset shape:", customer_data.shape)
print("\nSample data:")
print(customer_data.head())

# 2. Data preprocessing
print("\nData Summary:")
print(customer_data.describe())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# 3. Apply DBSCAN clustering
# We'll try different epsilon values to find the optimal clustering
eps_values = [0.3, 0.5, 0.7]
min_samples_values = [5, 10, 15]

best_silhouette = -1
best_eps = None
best_min_samples = None
best_labels = None

# Find optimal parameters
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Skip evaluation if all points are noise (-1)
        if len(set(labels)) <= 1 or -1 in labels and len(set(labels)) <= 2:
            continue
        
        # Calculate silhouette score (excluding noise points)
        mask = labels != -1
        if sum(mask) > 1 and len(set(labels[mask])) > 1:
            s_score = silhouette_score(X_scaled[mask], labels[mask])
            
            print(f"eps={eps}, min_samples={min_samples}: clusters={len(set(labels)) - (1 if -1 in labels else 0)}, "
                  f"noise points={sum(labels == -1)}, silhouette={s_score:.3f}")
            
            if s_score > best_silhouette:
                best_silhouette = s_score
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels

# Use the best parameters for final clustering
print(f"\nBest parameters: eps={best_eps}, min_samples={best_min_samples}, silhouette={best_silhouette:.3f}")

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to the original data
customer_data['Cluster'] = labels

# 4. Analyze clusters
print("\nCluster distribution:")
cluster_counts = customer_data['Cluster'].value_counts().sort_index()
print(cluster_counts)

# Calculate cluster statistics
print("\nCluster profiles:")
cluster_stats = customer_data.groupby('Cluster').mean()
print(cluster_stats)

# 5. Visualize results
# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))

# Create a scatter plot
colors = sns.color_palette("hls", len(set(labels)))
color_map = {i: colors[i] for i in range(len(colors))}
color_map[-1] = (0.5, 0.5, 0.5)  # Gray for noise points

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[color_map[label] for label in labels], 
            s=50, alpha=0.8, edgecolor='k')

# Add cluster centers if available
for cluster in sorted(set(labels)):
    if cluster != -1:  # Skip noise points
        cluster_center = X_pca[labels == cluster].mean(axis=0)
        plt.scatter(cluster_center[0], cluster_center[1], s=200, marker='X', 
                   color='black', edgecolor='w')
        plt.annotate(f'Cluster {cluster}', xy=cluster_center, 
                    xytext=(cluster_center[0] + 0.1, cluster_center[1] + 0.1),
                    fontsize=12, fontweight='bold')

plt.title('DBSCAN Clustering of Customer Data (PCA Projection)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(alpha=0.3)

# Create legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], 
                              markersize=10, label=f'Cluster {i}' if i != -1 else 'Noise')
                  for i in sorted(set(labels))]
plt.legend(handles=legend_elements, loc='best', title='Clusters')

plt.tight_layout()
plt.savefig('dbscan_customer_segmentation.png', dpi=300)
plt.show()

# 6. Feature importance visualization
plt.figure(figsize=(14, 10))

# Create a heatmap of feature means by cluster
pivot = pd.DataFrame(scaler.inverse_transform(X_scaled), 
                    columns=customer_data.columns[:-1])
pivot['Cluster'] = labels
pivot_table = pivot.groupby('Cluster').mean()

# Normalize the heatmap data for better visualization
norm_data = (pivot_table - pivot_table.min()) / (pivot_table.max() - pivot_table.min())

sns.heatmap(norm_data, annot=pivot_table.round(1), cmap='YlGnBu', fmt='.1f', linewidths=0.5)
plt.title('Normalized Feature Values by Cluster', fontsize=16)
plt.tight_layout()
plt.savefig('dbscan_cluster_features.png', dpi=300)
plt.show()

# 7. Customer profiling based on clusters
def profile_clusters(df):
    """Generate descriptive profiles for each cluster."""
    profiles = {}
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        
        if cluster == -1:
            profiles[cluster] = "Outliers/Noise: Customers with unusual behavior patterns"
            continue
            
        profile = "Cluster {}: ".format(cluster)
        
        # Income level
        mean_income = cluster_data['Income'].mean()
        if mean_income < 60:
            profile += "Lower income segment"
        elif mean_income < 120:
            profile += "Middle income segment"
        else:
            profile += "High income segment"
            
        # Spending behavior
        mean_purchase = cluster_data['PurchaseAmount'].mean()
        if mean_purchase < 40:
            profile += ", low spenders"
        elif mean_purchase < 100:
            profile += ", moderate spenders"
        else:
            profile += ", high spenders"
            
        # Digital engagement
        mean_web = cluster_data['WebVisits'].mean()
        if mean_web < 5:
            profile += ", low digital engagement"
        elif mean_web < 15:
            profile += ", moderate digital engagement"
        else:
            profile += ", high digital engagement"
            
        # In-store preference
        mean_store = cluster_data['StoreVisits'].mean()
        if mean_store < 2:
            profile += ", prefers online shopping"
        elif mean_store < 7:
            profile += ", balanced shopper"
        else:
            profile += ", prefers in-store shopping"
            
        profiles[cluster] = profile
        
    return profiles

# Generate and print cluster profiles
print("\nCustomer segment profiles:")
profiles = profile_clusters(customer_data)
for cluster, profile in profiles.items():
    print(f"{profile}")
    if cluster != -1:
        cluster_size = sum(customer_data['Cluster'] == cluster)
        cluster_percent = cluster_size / len(customer_data) * 100
        print(f"  - {cluster_size} customers ({cluster_percent:.1f}% of total)")
        
# 8. Additional analysis - comparing spending to income ratio
customer_data['SpendToIncomeRatio'] = customer_data['PurchaseAmount'] / customer_data['Income']

plt.figure(figsize=(12, 8))
for cluster in sorted(set(labels)):
    if cluster == -1:
        plt.scatter(customer_data[labels == cluster]['Income'], 
                   customer_data[labels == cluster]['PurchaseAmount'],
                   alpha=0.5, color='gray', label='Noise')
    else:
        plt.scatter(customer_data[labels == cluster]['Income'], 
                   customer_data[labels == cluster]['PurchaseAmount'],
                   alpha=0.7, label=f'Cluster {cluster}')

plt.title('Income vs. Purchase Amount by Cluster', fontsize=16)
plt.xlabel('Annual Income (thousands)', fontsize=12)
plt.ylabel('Annual Purchase Amount (hundreds)', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('income_vs_purchase.png', dpi=300)
plt.show()

# 9. Conclusion and recommendations
print("\nConclusion and Business Recommendations:")
print("Based on the DBSCAN clustering analysis, we identified distinct customer segments.")
print("These insights can be used to develop targeted marketing strategies:")

for cluster in sorted(set(labels)):
    if cluster != -1:  # Skip noise points
        cluster_data = customer_data[customer_data['Cluster'] == cluster]
        
        print(f"\nFor Cluster {cluster}:")
        print(f"- {profiles[cluster]}")
        
        # Generate specific recommendations based on cluster characteristics
        if cluster_data['Income'].mean() > 120 and cluster_data['PurchaseAmount'].mean() > 100:
            print("  * Target with premium products and loyalty rewards")
            print("  * Focus on exclusive experiences and personalized service")
        
        if cluster_data['WebVisits'].mean() > 15:
            print("  * Prioritize digital marketing channels and online experiences")
            print("  * Develop mobile app features for this digitally-engaged segment")
            
        if cluster_data['StoreVisits'].mean() > 7:
            print("  * Enhance in-store experience with personalized assistance")
            print("  * Implement in-store exclusive promotions")
            
        if cluster_data['SpendToIncomeRatio'].mean() > 1.0:
            print("  * High-value customers with high spend-to-income ratio")
            print("  * Focus on retention strategies and premium memberships")
        elif cluster_data['SpendToIncomeRatio'].mean() < 0.4:
            print("  * Potential to increase share of wallet")
            print("  * Consider targeted promotions to encourage higher spending")