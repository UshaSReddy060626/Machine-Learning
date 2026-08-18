from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42)
rf = RandomForestClassifier(n_estimators=100,
    criterion="gini",
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
print("DECISION TREE")
print("--------------------")
print("Training Accuracy:", dt.score(X_train, y_train))
print("Testing Accuracy:", dt.score(X_test, y_test))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))
print("F1 Score:", f1_score(y_test, dt_pred))
print("\nRANDOM FOREST")
print("--------------------")
print("Training Accuracy:", rf.score(X_train, y_train))
print("Testing Accuracy:", rf.score(X_test, y_test))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))