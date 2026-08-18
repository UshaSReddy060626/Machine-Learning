# 🌳 Decision Tree & 🌲 Random Forest

Today I learned and implemented **Decision Tree** and **Random Forest** classification algorithms using Scikit-Learn.

The examples cover model training, hyperparameter tuning, evaluation, feature importance, predictions, and overfitting checks.

---

## 🌳 Decision Tree

A **Decision Tree** makes predictions by splitting the dataset into branches based on feature values.

### Topics Covered

* Working with `fit()` and `predict()`
* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* Training accuracy
* Testing accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Feature Importance
* New Predictions
* Overfitting Check

### Important Hyperparameters

| Parameter           | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `criterion`         | Determines how the best split is selected   |
| `max_depth`         | Controls the maximum depth of the tree      |
| `min_samples_split` | Minimum samples required to split a node    |
| `min_samples_leaf`  | Minimum samples required in a leaf          |
| `max_features`      | Number of features considered for splitting |

---

## 🌲 Random Forest

A **Random Forest** combines multiple Decision Trees and uses their combined predictions to improve generalization.

### Topics Covered

* Multiple Decision Trees
* `n_estimators`
* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* Training accuracy
* Testing accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Feature Importance
* New Predictions
* Overfitting Check
* Comparison with Decision Tree

### Important Hyperparameters

| Parameter           | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `n_estimators`      | Number of trees in the forest              |
| `criterion`         | Determines how splits are evaluated        |
| `max_depth`         | Controls tree depth                        |
| `min_samples_split` | Minimum samples required for splitting     |
| `min_samples_leaf`  | Minimum samples required in a leaf         |
| `max_features`      | Number of features considered by each tree |

---

## 📊 Model Evaluation

Both models were evaluated using:

* **Training Accuracy**
* **Testing Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

These metrics help determine how well the models perform on both training and unseen data.

---

## ⚠️ Overfitting Check

Training and testing performance were compared to identify overfitting.

A large difference between training and testing accuracy can indicate that the model is learning the training data too closely.

Parameters such as:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

can be used to control model complexity.

---

## ⭐ Feature Importance

Both Decision Tree and Random Forest provide **feature importance**, which shows how useful each feature was for making predictions.

```python
model.feature_importances_
```

---

## 🔮 New Predictions

After training the models, new unseen samples were passed to the trained models using:

```python
model.predict(new_data)
```

---

## ⚖️ Decision Tree vs Random Forest

| Decision Tree       | Random Forest                         |
| ------------------- | ------------------------------------- |
| Uses a single tree  | Uses multiple trees                   |
| Easier to interpret | More complex                          |
| Can easily overfit  | Usually more resistant to overfitting |
| Faster to train     | Can require more computation          |
| Simpler model       | Generally stronger ensemble model     |

---

## 🚫 Feature Scaling

**No feature scaling was used.**

Decision Trees and Random Forests are tree-based algorithms and generally **do not require StandardScaler or MinMaxScaler**, because their splitting decisions are based on feature thresholds rather than distances or feature magnitudes.

---

## 🛠️ Practice

Implemented and compared:

* Decision Tree Classifier
* Random Forest Classifier
* Hyperparameters
* Model evaluation
* Feature importance
* New predictions
* Overfitting checks
* Performance comparison

---

## 🧠 Key Takeaways

* Decision Trees make predictions using a sequence of feature-based splits.
* Random Forest combines multiple Decision Trees.
* `max_depth` helps control tree complexity.
* `min_samples_split` and `min_samples_leaf` help reduce overfitting.
* `n_estimators` controls the number of trees in Random Forest.
* Tree-based models generally do not require feature scaling.
* Training and testing performance should be compared to detect overfitting.
* Feature importance helps understand which features contribute most to predictions.
* Random Forest generally provides more robust predictions than a single Decision Tree.
