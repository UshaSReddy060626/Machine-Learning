import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
X = np.array([
    [22, 30, 24],
    [25, 35, 20],
    [28, 40, 18],
    [30, 45, 15],
    [32, 50, 12],
    [35, 55, 10],
    [38, 60, 8],
    [40, 65, 7],
    [42, 70, 6],
    [45, 75, 5],
    [48, 80, 4],
    [50, 85, 3],
    [52, 90, 2],
    [55, 95, 1],
    [58, 100, 1],
    [60, 105, 1]
])
y = np.array([
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 1, 1,
    1, 1, 1, 1
])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
   stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs"
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
print(precision_score(y_test, y_pred))
print("\nRecall:")
print(recall_score(y_test, y_pred))
print("\nF1 Score:")
print(f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nCoefficients:")
print(model.coef_)
print("\nIntercept:")
print(model.intercept_)
new_customer = np.array([
    [45, 80, 3]
])
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)
print("\nNew Customer Prediction:")
if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")
print("Probability:")
print(probability)
