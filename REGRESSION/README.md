# 📈 Regression

Regression is a type of **Supervised Machine Learning** used to predict continuous numerical values.

Examples:

* House prices
* Salary
* Temperature
* Sales
* Exam scores

---

## 📚 Topics Covered

* Linear Regression
* Cost Function
* Linear Regression Assumptions
* Regression Evaluation Metrics
* Hands-on Linear Regression Implementation

---

## 1. 📈 Linear Regression

Linear Regression tries to find the best-fitting straight line between input features and a continuous target.

### Equation

```text
ŷ = β₀ + β₁x
```

Where:

* `ŷ` → Predicted value
* `β₀` → Intercept
* `β₁` → Slope/Coefficient
* `x` → Input feature

### Example

```text
Hours Studied → Exam Score
```

The model learns the relationship between study hours and exam scores and uses it to predict the score for new students.

---

## 2. 💰 Cost Function

A cost function measures how far the model's predictions are from the actual values.

One commonly used loss for Linear Regression is **Mean Squared Error (MSE)**.

### Formula

```text
MSE = (1/n) Σ(y - ŷ)²
```

### Basic Process

```text
Actual Value
     ↓
Compare with Prediction
     ↓
Calculate Error
     ↓
Square Error
     ↓
Average Errors
     ↓
MSE
```

A lower MSE generally means smaller squared prediction errors.

---

## 3. 🏔️ Gradient Descent

Gradient Descent is an optimization method that can be used to find model parameters that minimize the cost function.

Think of it as trying to walk downhill until reaching a low point.

```text
High Cost
    ↓
Take a step toward lower cost
    ↓
Update parameters
    ↓
Repeat
    ↓
Lower Cost
```

---

## 4. 📋 Linear Regression Assumptions

Linear Regression works best when several assumptions approximately hold.

### Linearity

There should be a roughly linear relationship between predictors and the target.

### Independence

Observations should generally be independent.

### Homoscedasticity

The spread of residuals should be reasonably consistent across the range of predictions.

### No Strong Multicollinearity

Input features should not have problematic levels of strong linear correlation with each other.

### Residual Distribution

For some statistical inference tasks, residuals are expected to be approximately normally distributed.

---

## 5. 📊 Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

```text
MAE = average(|Actual - Predicted|)
```

Easy to interpret because it is in the same units as the target.

---

### MSE — Mean Squared Error

Measures the average squared prediction error.

```text
MSE = average((Actual - Predicted)²)
```

Large errors are penalized more heavily.

---

### RMSE — Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE is in the same units as the target, making it easier to interpret than MSE.

---

### R² Score

R² measures how much variation in the target is explained by the model relative to a baseline that predicts the target mean.

```text
R² = 1
```

represents a perfect fit.

```text
R² = 0
```

means the model performs like the mean-prediction baseline in the usual setup.

R² can also be negative when the model performs worse than that baseline.

⚠️ R² should not be interpreted as classification accuracy.

---

## 6. 💻 Hands-on Implementation

### Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

### Create Dataset

```python
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    35,
    42,
    50,
    58,
    65,
    72,
    78,
    85
])
```

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

### Create and Train Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

### Make Predictions

```python
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
```

### Check Model Parameters

```python
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
```

### Evaluate the Model

```python
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

### Visualize

```python
plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression")

plt.show()
```

### Predict New Data

```python
prediction = model.predict([[9]])

print("Predicted Score:", prediction[0])
```

---

## 🧠 Quick Revision

```text
Regression
    ↓
Predict Continuous Numerical Values
    ↓
Linear Regression
    ↓
Find Best-Fitting Line
    ↓
ŷ = β₀ + β₁x
    ↓
Minimize Prediction Error
    ↓
Evaluate Model
    ↓
MAE / MSE / RMSE / R²
```

---

## 🎯 Key Takeaways

* Regression predicts continuous numerical values.
* Linear Regression models the relationship using a straight line.
* The equation is `ŷ = β₀ + β₁x`.
* The cost/loss function measures prediction error.
* MSE squares errors and averages them.
* Gradient Descent can be used to minimize the loss.
* Linear Regression has important assumptions such as linearity, independence, homoscedasticity, and limited multicollinearity.
* MAE measures average absolute error.
* MSE penalizes larger errors more heavily.
* RMSE is expressed in the target's units.
* R² measures explained variation relative to a mean-prediction baseline.
* Always evaluate regression models on unseen test data.
---


