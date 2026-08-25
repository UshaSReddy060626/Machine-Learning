# ⚖️ Imbalanced Data

Imbalanced data occurs when the classes in a dataset are not represented equally. For example, in a fraud detection dataset, most transactions may be legitimate while only a small number are fraudulent.

If the dataset is highly imbalanced, a machine learning model may become biased toward the majority class and perform poorly on the minority class.

## 🎯 Why Handle Imbalanced Data?

Imbalanced datasets can cause:

* ❌ Poor detection of minority classes
* ⚠️ High accuracy but poor real-world performance
* ⚖️ Model bias toward the majority class
* 📉 Low recall for the minority class

Common techniques for handling imbalanced data include:

* 📈 Oversampling
* 📉 Undersampling
* 🔄 SMOTE
* ⚖️ Class Weighting

---

## 1. 📈 Oversampling

Oversampling increases the number of samples in the minority class.

Instead of removing majority-class samples, it adds more minority-class samples.

### 📌 Example

```text
Before:
Class 0 → 900 samples
Class 1 → 100 samples

After Oversampling:
Class 0 → 900 samples
Class 1 → 900 samples
```

### ✅ Advantages

* Keeps all majority-class data
* Helps the model learn the minority class
* Simple to implement

### ⚠️ Disadvantage

Random oversampling can duplicate existing samples, which may increase the risk of overfitting.

---

## 2. 📉 Undersampling

Undersampling reduces the number of samples in the majority class.

### 📌 Example

```text
Before:
Class 0 → 900 samples
Class 1 → 100 samples

After Undersampling:
Class 0 → 100 samples
Class 1 → 100 samples
```

### ✅ Advantages

* Reduces dataset size
* Faster model training
* Creates a balanced dataset

### ⚠️ Disadvantage

Some useful information from the majority class may be removed.

---

## 3. 🔄 SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** creates new synthetic samples for the minority class instead of simply duplicating existing samples.

SMOTE works by selecting a minority-class sample and creating new samples between it and its neighboring minority-class samples.

### 📌 Example

```text
Original minority samples:
A
B
C

SMOTE:
A + B → New synthetic sample
B + C → New synthetic sample
```

### ✅ Advantages

* Creates new minority samples
* Reduces simple duplication
* Can improve minority-class learning

### ⚠️ Important

SMOTE should be applied **only to the training data**, not before splitting the dataset.

### ✔️ Correct Approach

```text
Original Data
     ↓
Train-Test Split
     ↓
Training Data
     ↓
SMOTE
     ↓
Balanced Training Data
```

This prevents **data leakage**.

### 💻 Example

```python
from imblearn.over_sampling import SMOTE

smote=SMOTE(random_state=42)
X_train,y_train=smote.fit_resample(X_train,y_train)
```

---

## 4. ⚖️ Class Weighting

Class weighting gives more importance to the minority class during model training.

Instead of changing the dataset, the model assigns different weights to different classes.

For example:

```text
Class 0 → Weight = 1
Class 1 → Weight = 5
```

A mistake on Class 1 therefore has a larger effect on the model's loss.

Many Scikit-learn models support:

```python
class_weight="balanced"
```

### 💻 Example

```python
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(class_weight="balanced")
model.fit(X_train,y_train)
```

### ✅ Advantages

* Does not modify the original dataset
* No synthetic samples are created
* Useful for many classification algorithms

---

## 📊 Comparison

| Technique          | What it does                           | Main Advantage            | Main Disadvantage        |
| ------------------ | -------------------------------------- | ------------------------- | ------------------------ |
| 📈 Oversampling    | Adds minority samples                  | Keeps majority data       | Can cause overfitting    |
| 📉 Undersampling   | Removes majority samples               | Faster training           | May lose information     |
| 🔄 SMOTE           | Creates synthetic minority samples     | Avoids simple duplication | Can create noisy samples |
| ⚖️ Class Weighting | Gives minority class higher importance | No data modification      | Depends on model support |

---

## 🔑 Key Points

* 📈 **Oversampling** increases minority-class samples.
* 📉 **Undersampling** decreases majority-class samples.
* 🔄 **SMOTE** generates synthetic minority-class samples.
* ⚖️ **Class Weighting** gives more importance to minority classes.
* 🔒 SMOTE should be applied **after train-test splitting**.
* ❌ Do not evaluate a model only using accuracy on highly imbalanced datasets.
* 📊 Metrics such as **Precision, Recall, F1-score, and ROC-AUC** can provide a better evaluation.

## 🚀 Practical Workflow

```text
Dataset
   ↓
🔍 Check Class Distribution
   ↓
✂️ Train-Test Split
   ↓
⚖️ Handle Imbalance
   ├── 📈 Oversampling
   ├── 📉 Undersampling
   ├── 🔄 SMOTE
   └── ⚖️ Class Weighting
   ↓
🤖 Train Model
   ↓
📊 Evaluate Model
```
