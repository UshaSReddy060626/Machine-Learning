import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
colors = ['navy','turquoise','darkorange']
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y==i,0], X_pca[y==i,1], color=colors[i], label=target_name, alpha=0.8, edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.savefig("pca_iris.png", dpi=300, bbox_inches='tight')
plt.show()