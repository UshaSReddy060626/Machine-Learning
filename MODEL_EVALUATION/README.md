# 📊 Model Evaluation

Model evaluation measures how well a machine learning model performs on unseen data.

## Classification

### Accuracy

Percentage of correct predictions.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

### Precision

How many predicted positives are actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

How many actual positives were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Balance between Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC Curve

Plots **True Positive Rate vs False Positive Rate** at different thresholds.

### ROC-AUC

Measures how well the model separates positive and negative classes. Higher is generally better.

### Precision-Recall Curve

Shows the trade-off between Precision and Recall. Useful for imbalanced datasets.

---

## Regression

### MAE

Average absolute prediction error.

```text
MAE = mean(|y - ŷ|)
```

### MSE

Average squared prediction error. Penalizes large errors more.

```text
MSE = mean((y - ŷ)²)
```

### RMSE

Square root of MSE. Expressed in the same units as the target.

```text
RMSE = √MSE
```

### R² Score

Measures the proportion of variance explained by the model. Closer to `1` is generally better.

---

## Validation

### Cross Validation

Splits data into multiple folds and evaluates the model multiple times.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
```

### Stratified K-Fold

Like K-Fold, but preserves the class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## Quick Summary

| Type           | Metrics                                                 |
| -------------- | ------------------------------------------------------- |
| Classification | Accuracy, Precision, Recall, F1, ROC, ROC-AUC, PR Curve |
| Regression     | MAE, MSE, RMSE, R²                                      |
| Validation     | Cross Validation, Stratified K-Fold                     |
