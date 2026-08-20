# 🔵 Unsupervised Learning

This section covers important **Unsupervised Learning** techniques used to discover patterns, groups, and structure in data without using labeled target values.

## 📚 Topics Covered

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN
* PCA
* Choosing the appropriate unsupervised learning technique
* Feature Scaling
* Dimensionality Reduction
* Cluster Analysis

---

## 🟢 K-Means Clustering

K-Means is a clustering algorithm that divides data into a predefined number of clusters.

### Key Concepts

* Clustering
* Centroids
* Cluster Assignment
* Distance
* Iterative Optimization
* Elbow Method
* Inertia

### Important Parameters

| Parameter      | Purpose                            |
| -------------- | ---------------------------------- |
| `n_clusters`   | Number of clusters                 |
| `n_init`       | Number of centroid initializations |
| `random_state` | Controls randomness                |

### Workflow

```text
Data
 ↓
Feature Scaling
 ↓
Choose K
 ↓
Initialize Centroids
 ↓
Assign Points
 ↓
Update Centroids
 ↓
Repeat
 ↓
Final Clusters
```

### Elbow Method

The Elbow Method can be used to help determine a suitable number of clusters by comparing the model's inertia for different values of `K`.

---

## 🟠 Hierarchical Clustering

Hierarchical Clustering builds a hierarchy of clusters by progressively combining or dividing groups.

The common approach used here is **Agglomerative Clustering**.

### Key Concepts

* Hierarchical Clustering
* Agglomerative Clustering
* Cluster Merging
* Distance
* Dendrogram
* Linkage

### Important Parameters

| Parameter    | Purpose                              |
| ------------ | ------------------------------------ |
| `n_clusters` | Number of final clusters             |
| `linkage`    | Determines how clusters are combined |

Common linkage methods:

* `ward`
* `complete`
* `average`
* `single`

### Dendrogram

A dendrogram visually represents the hierarchy of cluster formation and can help determine an appropriate number of clusters.

---

## 🔵 DBSCAN

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

It groups points based on their density and can identify points that do not belong to any cluster.

### Key Concepts

* Density-Based Clustering
* Core Points
* Border Points
* Noise Points
* Outlier Detection

### Important Parameters

| Parameter     | Purpose                                                      |
| ------------- | ------------------------------------------------------------ |
| `eps`         | Defines the neighborhood radius                              |
| `min_samples` | Minimum number of nearby samples required for a dense region |

DBSCAN represents noise points using:

```text
-1
```

### Advantages

* Does not require the number of clusters beforehand
* Can identify noise and outliers
* Can find clusters with irregular shapes

---

## 🟣 PCA

PCA stands for **Principal Component Analysis**.

It is a dimensionality reduction technique that transforms a dataset into a smaller number of principal components while preserving as much variance as possible.

### Key Concepts

* Dimensionality Reduction
* Principal Components
* Variance
* Explained Variance
* Feature Transformation

### Important Parameter

| Parameter      | Purpose                                |
| -------------- | -------------------------------------- |
| `n_components` | Number of principal components to keep |

### PCA Workflow

```text
Original Data
 ↓
Feature Scaling
 ↓
PCA
 ↓
Principal Components
 ↓
Reduced-Dimensional Data
```

### Explained Variance

`explained_variance_ratio_` shows the proportion of variance captured by each principal component.

---

## ⚖️ Clustering Comparison

| Feature                     | K-Means      | Hierarchical            | DBSCAN               |
| --------------------------- | ------------ | ----------------------- | -------------------- |
| Requires number of clusters | Yes          | Usually                 | No                   |
| Uses centroids              | Yes          | No                      | No                   |
| Detects noise               | No           | No                      | Yes                  |
| Irregular cluster shapes    | Limited      | Possible                | Yes                  |
| Main parameter              | `n_clusters` | `n_clusters`, `linkage` | `eps`, `min_samples` |
| Visualization               | Cluster plot | Dendrogram              | Cluster plot         |

---

## 🔄 PCA vs Clustering

PCA and clustering solve different problems.

### Clustering

Answers:

> "Which data points belong together?"

Examples:

* K-Means
* Hierarchical Clustering
* DBSCAN

### PCA

Answers:

> "Can I represent this high-dimensional data using fewer dimensions?"

PCA can also be used before clustering:

```text
High-Dimensional Data
        ↓
   StandardScaler
        ↓
        PCA
        ↓
Reduced Data
        ↓
     K-Means
        ↓
    Clusters
```

---

## 🚫 Feature Scaling

Feature scaling is important for these techniques because many of them rely on distances or variance.

A common workflow is:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Scaling is especially important for:

* K-Means
* Hierarchical Clustering
* DBSCAN
* PCA

---

## 🛠️ Practice

Implemented examples for:

* K-Means clustering
* K-Means hyperparameters
* Elbow Method
* Hierarchical Clustering
* Agglomerative Clustering
* Dendrogram
* Linkage methods
* DBSCAN clustering
* `eps`
* `min_samples`
* Noise/outlier detection
* PCA dimensionality reduction
* `n_components`
* Explained variance
* PCA visualization
* Feature scaling
* Understanding when to use each technique

---

## 🧠 Key Takeaways

* Unsupervised Learning works without labeled target values.
* K-Means divides data into a predefined number of clusters.
* Centroids represent the center of K-Means clusters.
* The Elbow Method can help select a suitable value of `K`.
* Hierarchical Clustering creates a hierarchy of clusters.
* A dendrogram visualizes the hierarchy.
* DBSCAN groups points based on density.
* DBSCAN can detect noise and outliers.
* `eps` and `min_samples` are important DBSCAN parameters.
* PCA reduces the dimensionality of a dataset.
* `n_components` determines how many principal components are retained.
* Explained variance shows how much variation is captured by the components.
* Feature scaling is important for K-Means, Hierarchical Clustering, DBSCAN, and PCA.
* K-Means, Hierarchical Clustering, and DBSCAN are clustering techniques, while PCA is a dimensionality reduction technique.
