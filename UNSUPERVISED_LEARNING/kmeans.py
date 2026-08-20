import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.DataFrame({
    "age": [20, 22, 25, 27, 40, 42, 45, 48],
    "income": [20, 25, 28, 30, 70, 75, 80, 85]})
X = data[["age", "income"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10)
model.fit(X_scaled)
data["cluster"] = model.labels_
print(data)