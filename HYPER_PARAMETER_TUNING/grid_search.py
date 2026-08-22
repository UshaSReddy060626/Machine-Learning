from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    "max_depth": [2, 3, 5, 10],
    "min_samples_split": [2, 5, 10]}
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy")
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)