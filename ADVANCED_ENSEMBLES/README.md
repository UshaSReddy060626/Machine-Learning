# 🌳 Advanced Classification Models

This section covers **SVM, XGBoost, LightGBM, and CatBoost**, with a focus on understanding how they work, when to use them, important hyperparameters, and practical implementation using Python and Scikit-Learn-compatible APIs.

---

## 📚 Topics Covered

* Support Vector Machine (SVM)
* XGBoost
* LightGBM
* CatBoost
* When to use each model
* Important hyperparameters
* Model comparison
* Practical implementation

---

## 🟣 Support Vector Machine (SVM)

**Support Vector Machine** is a supervised learning algorithm that finds a decision boundary that separates different classes while maximizing the margin between them.

### Key Concepts

* Decision Boundary
* Hyperplane
* Margin
* Support Vectors
* Linear and Non-linear Classification
* Kernel Functions

### Important Hyperparameters

| Parameter | Purpose                                                              |
| --------- | -------------------------------------------------------------------- |
| `C`       | Controls the trade-off between margin size and classification errors |
| `kernel`  | Determines the type of decision boundary                             |
| `gamma`   | Controls the influence of individual training samples                |

Common kernels:

* `linear`
* `rbf`
* `poly`
* `sigmoid`

### Important Note

SVM generally benefits from **feature scaling**, unlike tree-based models.

---

## 🟠 XGBoost

**XGBoost (Extreme Gradient Boosting)** is a powerful gradient boosting algorithm that builds decision trees sequentially, with each new tree helping improve previous predictions.

### Key Concepts

* Gradient Boosting
* Sequential Decision Trees
* Learning Rate
* Regularization
* Ensemble Learning

### Important Hyperparameters

| Parameter          | Purpose                                |
| ------------------ | -------------------------------------- |
| `n_estimators`     | Number of boosting trees               |
| `learning_rate`    | Controls the contribution of each tree |
| `max_depth`        | Controls tree complexity               |
| `subsample`        | Fraction of training samples used      |
| `colsample_bytree` | Fraction of features used by each tree |

### When to Use

XGBoost is a strong general-purpose choice for **tabular datasets** and is widely used when high predictive performance is important.

---

## 🟢 LightGBM

**LightGBM** is a gradient boosting framework designed for high performance and efficient training, especially on large datasets.

### Key Concepts

* Gradient Boosting
* Histogram-based Learning
* Leaf-wise Tree Growth
* Efficient Training
* Large-Scale Tabular Data

### Important Hyperparameters

| Parameter          | Purpose                                 |
| ------------------ | --------------------------------------- |
| `n_estimators`     | Number of boosting trees                |
| `learning_rate`    | Controls the contribution of each tree  |
| `num_leaves`       | Controls the number of leaves in a tree |
| `max_depth`        | Controls maximum tree depth             |
| `subsample`        | Fraction of training samples used       |
| `colsample_bytree` | Fraction of features used               |

### When to Use

LightGBM is particularly useful when working with **large tabular datasets** where training speed and efficiency are important.

---

## 🔵 CatBoost

**CatBoost** is a gradient boosting algorithm designed especially to handle **categorical features** effectively.

### Key Concepts

* Gradient Boosting
* Categorical Features
* Ordered Boosting
* Automatic Categorical Feature Handling
* Regularization

### Important Hyperparameters

| Parameter       | Purpose                                     |
| --------------- | ------------------------------------------- |
| `iterations`    | Number of boosting iterations               |
| `learning_rate` | Controls the contribution of each iteration |
| `depth`         | Controls tree depth                         |
| `l2_leaf_reg`   | Controls L2 regularization                  |
| `loss_function` | Defines the optimization objective          |

### When to Use

CatBoost is particularly useful for **tabular datasets containing categorical features**, such as:

* City
* Product Category
* Gender
* Device Type
* Customer Segment

---

## ⚖️ Model Comparison

| Model    | Best Use Case                                           | Scaling          |
| -------- | ------------------------------------------------------- | ---------------- |
| SVM      | Smaller/medium datasets and complex decision boundaries | Usually required |
| XGBoost  | Strong general-purpose tabular model                    | Not required     |
| LightGBM | Large datasets and fast training                        | Not required     |
| CatBoost | Datasets with categorical features                      | Not required     |

---

## 🌲 Random Forest vs Boosting

### Random Forest

Random Forest builds multiple trees that are largely independent of one another and combines their predictions.

```text
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┼──> Final Prediction
Tree 4 ──┤
Tree 5 ──┘
```

### Boosting

Boosting builds trees sequentially, with later trees improving the errors of previous trees.

```text
Tree 1
   ↓
Previous Errors
   ↓
Tree 2
   ↓
Previous Errors
   ↓
Tree 3
   ↓
Final Prediction
```

### Key Difference

**Random Forest → many independent trees**

**Boosting → trees learn sequentially from previous errors**

---

## 🛠️ Practice

Implemented examples for:

* SVM classification
* SVM feature scaling
* SVM hyperparameters
* XGBoost classification
* XGBoost hyperparameters
* LightGBM classification
* LightGBM hyperparameters
* CatBoost classification
* Categorical feature handling
* Model selection based on dataset characteristics

---

## 🧠 Key Takeaways

* SVM finds a decision boundary with a large margin between classes.
* `C`, `kernel`, and `gamma` are important SVM hyperparameters.
* SVM generally benefits from feature scaling.
* XGBoost uses sequential decision trees to improve previous predictions.
* `n_estimators`, `learning_rate`, and `max_depth` are important XGBoost parameters.
* LightGBM focuses on efficient and fast gradient boosting, especially for large datasets.
* `num_leaves` is an important LightGBM parameter.
* CatBoost is especially useful when categorical features are present.
* `iterations`, `learning_rate`, and `depth` are important CatBoost parameters.
* Tree-based boosting models generally do not require feature scaling.
* Random Forest builds trees independently, while boosting builds trees sequentially.
* Choosing a model depends on the dataset size, feature types, computational requirements, and problem characteristics.
