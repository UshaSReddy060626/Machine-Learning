import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
data = pd.DataFrame({
    "age": [20, 22, 25, 27, 40, 42, 45, 48, 90],
    "income": [20, 25, 28, 30, 70, 75, 80, 85, 10]})
X = data[["age", "income"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = DBSCAN(
    eps=0.8,
    min_samples=2)
labels = model.fit_predict(X_scaled)
data["cluster"] = labels
print(data)