# 🛡️ Regularization

Regularization is a technique used to reduce **overfitting** and improve a model's ability to **generalize to unseen data**.

## 📚 Topics Covered

* 🔹 L1 Regularization
* 🔹 L2 Regularization
* 🔹 ElasticNet
* 📈 Learning Curves
* ⏹️ Early Stopping

---

## 🔹 L1 Regularization

L1 adds a penalty based on the **absolute values of coefficients**.

It can make some coefficients exactly **zero**, which helps with feature selection.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

**L1 → Lasso → Can make coefficients zero**

---

## 🔹 L2 Regularization

L2 penalizes large coefficients and **shrinks them toward zero**.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)
```

**L2 → Ridge → Shrinks coefficients**

---

## 🔹 ElasticNet

ElasticNet combines **L1 + L2 regularization**.

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

model.fit(X_train, y_train)
```

`l1_ratio` controls the balance:

```text
1   → L1
0   → L2
0.5 → L1 + L2
```

---

## 📈 Learning Curves

Learning Curves show how **training and validation performance** change as the amount of training data increases.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
```

They help identify:

* 🔍 Overfitting
* 🔍 Underfitting
* 📊 Whether more training data may help

---

## ⏹️ Early Stopping

Early Stopping stops training when **validation performance stops improving**.

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10
)

model.fit(X_train, y_train)
```

It can help:

* 🛡️ Reduce overfitting
* ⏱️ Save training time
* 🎯 Prevent unnecessary training

---

## ⚖️ L1 vs L2 vs ElasticNet

| Technique     | Main Idea                  |
| ------------- | -------------------------- |
| 🔹 L1         | Can make coefficients zero |
| 🔹 L2         | Shrinks coefficients       |
| 🔹 ElasticNet | Combines L1 and L2         |

---

## 🧠 Quick Summary

```text
L1
→ Feature selection
→ Can make coefficients zero

L2
→ Shrinks coefficients
→ Controls model complexity

ElasticNet
→ L1 + L2

Learning Curves
→ Training vs Validation performance

Early Stopping
→ Stops training when validation performance stops improving
```

## 🎯 Main Goal

```text
Regularization
      ↓
🛡️ Reduce Overfitting
      ↓
📈 Improve Generalization
```
