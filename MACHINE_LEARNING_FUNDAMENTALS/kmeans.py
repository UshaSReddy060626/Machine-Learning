import numpy as np
from sklearn.cluster import KMeans

# [Annual spending, Number of visits]
X = np.array([[10, 2],[12, 3],[11, 2],[50, 10],[55, 11],[52, 10]])

# Create model
model = KMeans(n_clusters=2, random_state=42, n_init=10)

# Find groups
model.fit(X)

# Get cluster assigned to each customer
clusters = model.labels_
print("Clusters:", clusters)