from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("DECISION TREE")
print("--------------------")
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nFeature Importance:")
for feature, importance in zip(data.feature_names, model.feature_importances_):
    print(feature, ":", round(importance, 4))
new_data = [X_test[0]]
prediction = model.predict(new_data)
print("\nNew Prediction:", data.target_names[prediction[0]])