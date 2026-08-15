from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
iris = load_iris()
X = iris.data

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 4 features → 2 features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Original shape:", X.shape)
print("New shape:", X_pca.shape)