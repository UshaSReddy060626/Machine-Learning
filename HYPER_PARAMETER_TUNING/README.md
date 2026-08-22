# 🎯 Hyperparameter Tuning

Hyperparameter tuning is the process of finding good hyperparameter values for a machine learning model.

## 📚 Topics Covered

* Hyperparameters
* Grid Search
* Random Search
* Cross Validation with tuning
* Grid Search vs Random Search

---

## 🔹 Hyperparameters

Hyperparameters are settings chosen before training the model.

Example:

```python
DecisionTreeClassifier(max_depth=5)
```

Here, `max_depth` is a hyperparameter.

### Parameter vs Hyperparameter

```text
Parameter → Learned by the model from data

Hyperparameter → Set by us
```

---

## 🔹 Grid Search

Grid Search tests **every possible combination** of the given hyperparameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [2, 3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
```

### Advantages

* Exhaustive search
* Easy to understand
* Good for small parameter spaces

### Disadvantage

* Can be slow with many combinations

---

## 🔹 Random Search

Random Search randomly selects a specified number of hyperparameter combinations.

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    model,
    param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)
```

### Advantages

* Faster for large parameter spaces
* Tests different combinations efficiently

### Disadvantage

* May not test the best possible combination

---

## 🔹 Grid Search vs Random Search

| Grid Search                  | Random Search                   |
| ---------------------------- | ------------------------------- |
| Tests every combination      | Tests random combinations       |
| More exhaustive              | Faster                          |
| Can be expensive             | More efficient for large spaces |
| Best for small search spaces | Best for large search spaces    |

---

## 🧠 Key Takeaways

```text
Hyperparameter Tuning
        ↓
Find good model settings

Grid Search
        ↓
Try every combination

Random Search
        ↓
Try selected random combinations
```

### Rule of Thumb

```text
Small parameter space → Grid Search

Large parameter space → Random Search
```

Hyperparameter tuning helps improve model performance while avoiding unnecessary trial and error.
