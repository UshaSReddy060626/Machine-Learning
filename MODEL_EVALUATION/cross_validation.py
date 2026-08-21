from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
data = load_breast_cancer()
X = data.data
y = data.target
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42)
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy")
print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())