import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset correctly
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target_names[data.target], name='Species')

print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Features: {', '.join(X.columns)}")
print(f"Target classes: {', '.join(np.unique(y))}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Map species to numeric for easier comparison
species_mapping = {name: i for i, name in enumerate(y.unique())}
y_numeric = y.map(species_mapping)

# Rest of the code remains the same as in the previous version
# (The rest of the implementation would follow the same structure)