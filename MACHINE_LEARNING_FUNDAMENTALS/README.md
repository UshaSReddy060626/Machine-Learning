# 🤖 Machine Learning Fundamentals

This section covers the fundamental concepts required to understand Machine Learning algorithms and how ML models are built and evaluated.

---

## 📚 Topics Covered

* Machine Learning Pipeline
* Supervised Learning
* Unsupervised Learning
* Reinforcement Learning
* Overfitting
* Underfitting
* Bias vs Variance
* Generalization

---

## 🔄 Machine Learning Pipeline

A typical Machine Learning workflow is:

```text
Data Collection
      ↓
EDA
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Improvement
      ↓
Deployment
      ↓
Monitoring
```

---

## 👨‍🏫 Supervised Learning

Supervised Learning uses data where the correct answers are already available.

```text
Input Data + Correct Answers
          ↓
       ML Model
          ↓
      Prediction
```

### Types

#### Regression

Used to predict numerical values.

Examples:

* House prices
* Salary
* Temperature
* Sales

#### Classification

Used to predict categories.

Examples:

* Spam / Not Spam
* Cat / Dog
* Fraud / Normal
* Pass / Fail

---

## 🔍 Unsupervised Learning

Unsupervised Learning works with data where predefined answers or labels are not available.

The model attempts to discover patterns or groups within the data.

### Common Applications

* Customer segmentation
* Clustering
* Pattern discovery
* Dimensionality reduction

### Examples

* K-Means Clustering
* PCA

---

## 🎮 Reinforcement Learning

Reinforcement Learning is based on an agent interacting with an environment and learning through rewards and penalties.

```text
Agent
  ↓
Action
  ↓
Environment
  ↓
Reward / Penalty
  ↓
Learning
```

### Important Terms

| Term        | Meaning                           |
| ----------- | --------------------------------- |
| Agent       | Learner                           |
| Environment | World in which the agent operates |
| State       | Current situation                 |
| Action      | Decision taken by the agent       |
| Reward      | Feedback received                 |
| Policy      | Strategy used to choose actions   |

---

## 😵 Overfitting

Overfitting occurs when a model learns the training data too closely, including noise and unnecessary details.

Typical pattern:

```text
Training Performance → Very High
Testing Performance  → Poor
```

### Main Problem

The model does not generalize well to unseen data.

### Associated Concept

```text
High Variance → Overfitting
```

---

## 🥱 Underfitting

Underfitting occurs when a model is too simple to learn the important patterns in the data.

Typical pattern:

```text
Training Performance → Poor
Testing Performance  → Poor
```

### Main Problem

The model has not learned enough from the data.

### Associated Concept

```text
High Bias → Underfitting
```

---

## ⚖️ Bias vs Variance

### Bias

Bias represents systematic error caused by overly simple assumptions.

```text
High Bias
    ↓
Model too simple
    ↓
Underfitting
```

### Variance

Variance represents how sensitive the model is to changes in the training data.

```text
High Variance
      ↓
Model too complex
      ↓
Overfitting
```

### Goal

The goal is to find a good balance between bias and variance so that the model generalizes well.

```text
Low enough Bias
       +
Low enough Variance
       ↓
Good Generalization
```

---

## 🧠 Generalization

Generalization means that a Machine Learning model performs well not only on training data but also on new, unseen data.

```text
Training Data
      ↓
Learn Useful Patterns
      ↓
New Unseen Data
      ↓
Good Predictions
```

A good ML model should learn **patterns**, not simply memorize the training data.

---

## 📊 Quick Comparison

| Concept                | Main Idea                            |
| ---------------------- | ------------------------------------ |
| Supervised Learning    | Learn from labeled data              |
| Unsupervised Learning  | Find patterns in unlabeled data      |
| Reinforcement Learning | Learn through rewards and penalties  |
| Regression             | Predict numbers                      |
| Classification         | Predict categories                   |
| Overfitting            | Learns too much / memorizes          |
| Underfitting           | Learns too little                    |
| High Bias              | Usually associated with underfitting |
| High Variance          | Usually associated with overfitting  |
| Generalization         | Perform well on unseen data          |

---


