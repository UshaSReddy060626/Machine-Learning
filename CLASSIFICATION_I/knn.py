from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Predictions:")
print(y_pred)
print("\nActual:")
print(y_test)
probabilities = model.predict_proba(X_test_scaled)
print("\nPrediction Probabilities:")
print(probabilities)
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))
print("\nPrecision:")
print(precision_score(y_test, y_pred, average="weighted"))
print("\nRecall:")
print(recall_score(y_test, y_pred, average="weighted"))
print("\nF1 Score:")
print(f1_score(y_test, y_pred, average="weighted"))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="euclidean"
    )
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_test_scaled)
    score = accuracy_score(y_test, predictions)
    print(f"k = {k}, Accuracy = {score:.2f}")
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
probability = model.predict_proba(new_flower_scaled)
print("\nNew Flower:")
print("Predicted Class:", iris.target_names[prediction[0]])
print("\nProbability:")
print(probability)
