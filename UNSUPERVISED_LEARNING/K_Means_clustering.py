import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    'Age': [20, 22, 23, 25, 27, 30, 35, 40, 45, 50, 55, 60, 62, 65],
    'Income': [20, 22, 25, 24, 28, 45, 48, 50, 53, 52, 55, 70, 72, 75]}
df = pd.DataFrame(data)
df
plt.scatter(df['Age'], df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Scatter Plot of Age vs Income')
plt.show()
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(df[['Age', 'Income']])
df
plt.scatter(df['Age'], df['Income'], c=df['cluster'], cmap='rainbow')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Scatter Plot of Age vs Income')
plt.show()
centers = model.cluster_centers_
plt.scatter(df['Age'], df['Income'], c=df['cluster'], cmap='rainbow')
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Scatter Plot of Age vs Income with Clusters and Centroids')
plt.show()