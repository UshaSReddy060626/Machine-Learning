import pandas as pd
from catboost import CatBoostClassifier
data = pd.DataFrame({
    "age": [20, 25, 30, 35, 40, 45],
    "city": ["Bangalore", "Delhi", "Mumbai", "Delhi", "Bangalore", "Mumbai"],
    "bought": [0, 0, 1, 1, 1, 1]})
X = data[["age", "city"]]
y = data["bought"]
model = CatBoostClassifier(iterations=100,learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    verbose=False,
    random_seed=42)
model.fit(X, y, cat_features=["city"])
new_data = pd.DataFrame({
    "age": [28],
    "city": ["Bangalore"]})
prediction = model.predict(new_data)
print("Prediction:", prediction)