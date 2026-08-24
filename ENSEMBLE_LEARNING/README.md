# 🤝 Ensemble Learning

Ensemble Learning combines multiple machine learning models to make better and more reliable predictions.

## 📚 Topics Covered

* 👜 Bagging
* 🚀 Boosting
* 🧩 Stacking
* 🗳️ Voting Classifier
* 🗳️ Hard Voting
* 🎯 Soft Voting

---

## 👜 Bagging

Bagging trains multiple models independently using different samples of the training data and combines their predictions.

**Example:** Random Forest

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
```

**Bagging → Independent models → Combine predictions**

---

## 🚀 Boosting

Boosting trains models sequentially, where each new model focuses on improving the mistakes of previous models.

**Examples:**

* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

```python
from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```

**Boosting → Learn from previous mistakes**

---

## 🧩 Stacking

Stacking combines predictions from multiple models and passes them to another model called a **meta-model**.

```text
Model 1 ──┐
Model 2 ──┼──→ Meta Model → Final Prediction
Model 3 ──┘
```

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

model=StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression()
)
```

**Stacking → Base Models + Meta Model**

---

## 🗳️ Voting Classifier

Voting combines predictions from multiple classification models.

### Hard Voting

The class receiving the most votes becomes the final prediction.

```python
model=VotingClassifier(
    estimators=models,
    voting="hard"
)
```

**Hard Voting → Majority class wins**

### Soft Voting

Soft Voting uses the predicted probabilities from each model.

```python
model=VotingClassifier(
    estimators=models,
    voting="soft"
)
```

**Soft Voting → Highest combined probability wins**

---

## ⚖️ Comparison

| Method      | Main Idea                    | Example            |
| ----------- | ---------------------------- | ------------------ |
| 👜 Bagging  | Independent models           | Random Forest      |
| 🚀 Boosting | Learn from previous mistakes | XGBoost            |
| 🧩 Stacking | Base models + meta-model     | StackingClassifier |
| 🗳️ Voting  | Models vote                  | VotingClassifier   |

---

## 🧠 Quick Summary

```text
Bagging
→ Train models independently
→ Combine predictions

Boosting
→ Train models sequentially
→ Correct previous mistakes

Stacking
→ Multiple models
→ Meta-model makes final prediction

Voting
→ Multiple models vote
→ Majority/probability determines prediction
```

## 🎯 Main Goal

```text
Multiple Models
      ↓
Combine Their Strengths
      ↓
Better Predictions
      ↓
More Robust Model
```
