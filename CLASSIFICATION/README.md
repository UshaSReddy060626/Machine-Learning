# 🎯 Classification I

Classification is a type of **Supervised Machine Learning** where the model predicts a category or class.

Examples:

```text
Spam / Not Spam
Pass / Fail
Cat / Dog
Fraud / Normal
```

---

## 📚 Algorithms Covered

* Logistic Regression
* k-Nearest Neighbors (k-NN)
* Naive Bayes

For each algorithm:

* Working
* Hyperparameters
* Pros and Cons
* Use Cases
* Hands-on implementation

---

# 1. 🔵 Logistic Regression

Logistic Regression is a classification algorithm that predicts the probability of an observation belonging to a class.

Although its name contains "Regression", it is commonly used for classification.

## ⚙️ Working

```text
Input Features
      ↓
Weighted Combination
      ↓
Sigmoid Function
      ↓
Probability 0 to 1
      ↓
Classification Threshold
      ↓
Predicted Class
```

For example:

```text
Probability = 0.85
Threshold = 0.50

0.85 > 0.50
     ↓
Class 1
```

## 💻 Implementation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data[:100]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction probabilities
probabilities = model.predict_proba(X_test)

print("Probabilities:")
print(probabilities)
```

## ⚙️ Important Hyperparameters

### `C`

Controls the strength of regularization.

```python
LogisticRegression(C=1.0)
```

### `max_iter`

Maximum number of optimization iterations.

```python
LogisticRegression(max_iter=1000)
```

### `solver`

Optimization algorithm used by the model.

```python
LogisticRegression(solver="lbfgs")
```

## ✅ Pros

* Simple
* Fast
* Easy to interpret
* Produces probabilities
* Good baseline classifier

## ❌ Cons

* Limited for complex nonlinear patterns
* Often benefits from feature scaling
* Multicollinearity can affect coefficient interpretation

## 🎯 Use Cases

* Spam detection
* Customer churn
* Fraud detection
* Disease classification
* Pass/fail prediction

---

# 2. 🟢 k-Nearest Neighbors (k-NN)

k-NN predicts the class of a new data point using the classes of its nearest neighbors.

## ⚙️ Working

```text
New Data Point
      ↓
Calculate Distances
      ↓
Find k Nearest Neighbors
      ↓
Majority Vote
      ↓
Prediction
```

For example, if:

```text
k = 5

Neighbor 1 → Cat
Neighbor 2 → Cat
Neighbor 3 → Dog
Neighbor 4 → Cat
Neighbor 5 → Dog
```

The prediction is:

```text
Cat
```

because Cat has the majority vote.

## 💻 Implementation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## ⚙️ Important Hyperparameters

### `n_neighbors`

Number of neighbors used for prediction.

```python
KNeighborsClassifier(n_neighbors=5)
```

### `weights`

Controls the importance of neighbors.

```python
weights="uniform"
```

or:

```python
weights="distance"
```

### `metric`

Controls how distance is measured.

```python
metric="euclidean"
```

## ⚠️ Feature Scaling

Because k-NN depends on distance, feature scaling is usually important.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## ✅ Pros

* Simple to understand
* Easy to implement
* Can model nonlinear patterns
* No complicated training procedure

## ❌ Cons

* Prediction can be slow on large datasets
* Sensitive to feature scaling
* Sensitive to irrelevant features
* Choosing `k` is important
* Can require significant memory

## 🎯 Use Cases

* Pattern recognition
* Similarity-based classification
* Small classification datasets
* Image-related classification tasks

---

# 3. 🟣 Naive Bayes

Naive Bayes is a probability-based classification algorithm based on **Bayes' theorem**.

## ⚙️ Working

The algorithm starts with probabilities and updates them based on observed evidence.

```text
Prior Probability
       ↓
Observed Features
       ↓
Bayesian Update
       ↓
Class Probability
       ↓
Prediction
```

### Key Assumption

Naive Bayes assumes that features are **conditionally independent given the class**.

This assumption is often unrealistic, but the algorithm can still work very well in many applications.

## 💻 Gaussian Naive Bayes

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📝 Naive Bayes for Text

Naive Bayes is especially useful for text classification.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "win free money",
    "claim your prize",
    "meeting at 5 pm",
    "please send the report",
    "free prize winner",
    "project meeting tomorrow"
]

labels = [
    "spam",
    "spam",
    "normal",
    "normal",
    "spam",
    "normal"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)

model = MultinomialNB()

model.fit(X, labels)

new_message = ["free money prize"]

new_X = vectorizer.transform(new_message)

prediction = model.predict(new_X)

print("Prediction:", prediction[0])
```

## ⚙️ Important Hyperparameters

For `GaussianNB`:

```python
GaussianNB(var_smoothing=1e-9)
```

For text classification, the appropriate Naive Bayes variant depends on how the text is represented.

### Common Variants

| Variant       | Common Use                    |
| ------------- | ----------------------------- |
| GaussianNB    | Continuous numerical features |
| MultinomialNB | Count-based text features     |
| BernoulliNB   | Binary features               |

## ✅ Pros

* Fast
* Simple
* Works well with high-dimensional data
* Good for text classification
* Requires relatively little training data

## ❌ Cons

* Strong independence assumption
* May perform poorly when features are strongly dependent
* Different variants are suited to different feature types

## 🎯 Use Cases

* Spam detection
* Sentiment analysis
* Document classification
* News classification
* Text categorization

---

# 🆚 Algorithm Comparison

| Feature             | Logistic Regression                 | k-NN                     | Naive Bayes            |
| ------------------- | ----------------------------------- | ------------------------ | ---------------------- |
| Main idea           | Probability-based linear classifier | Nearest-neighbor voting  | Bayesian probability   |
| Training            | Relatively fast                     | Very little fitting      | Very fast              |
| Prediction          | Fast                                | Can be slower            | Fast                   |
| Scaling             | Often useful                        | **Very important**       | Depends on variant     |
| Nonlinear patterns  | Limited                             | Yes                      | Depends on data        |
| Text classification | Good                                | Usually not first choice | **Excellent baseline** |
| Main hyperparameter | `C`                                 | `n_neighbors`            | `var_smoothing`        |
| Easy to understand  | ⭐⭐⭐⭐                                | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐                   |

---

# 🧠 Quick Revision

### Logistic Regression

```text
Features
 ↓
Sigmoid
 ↓
Probability
 ↓
Threshold
 ↓
Class
```

### k-NN

```text
New Point
 ↓
Find Neighbors
 ↓
Majority Vote
 ↓
Class
```

### Naive Bayes

```text
Prior Probability
 ↓
Evidence
 ↓
Bayesian Update
 ↓
Class Probability
 ↓
Class
```

---

