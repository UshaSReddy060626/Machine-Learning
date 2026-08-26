# 🧠 Explainable AI

Explainable AI (XAI) refers to techniques that help us understand **how and why a machine learning model makes its predictions**.

Machine learning models can sometimes behave like black boxes. XAI helps make these predictions more transparent and interpretable.

## 🎯 Why Explainable AI?

XAI helps with:

* 🔍 Understanding model predictions
* 📊 Identifying important features
* 🐛 Detecting unexpected model behavior
* 🤝 Building trust in ML systems
* ⚖️ Supporting fairness and accountability
* 🏭 Improving real-world model deployment

---

## 📊 1. Feature Importance

Feature importance tells us which features have the greatest influence on a model's predictions.

For example:

```text
Contract Length → 0.35
Tenure          → 0.27
Monthly Charges → 0.20
Support Calls   → 0.12
Age             → 0.06
```

### Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data=load_breast_cancer()

X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

model=RandomForestClassifier(random_state=42)
model.fit(X,y)

importance=pd.Series(model.feature_importances_,index=X.columns)

print(importance.sort_values(ascending=False).head(10))
```

### Visualization

```python
import matplotlib.pyplot as plt

importance.sort_values(ascending=True).tail(10).plot(kind="barh")

plt.xlabel("Importance")
plt.title("Top Feature Importances")
plt.show()
```

### Limitation

Feature importance provides an overall view of feature influence but does not explain exactly why a particular prediction was made.

---

## 🔍 2. SHAP

**SHAP (SHapley Additive exPlanations)** explains how individual features contribute to model predictions.

SHAP is based on **Shapley values** from cooperative game theory.

A feature can have:

* ➕ Positive contribution
* ➖ Negative contribution
* 0️⃣ Little or no contribution

### Installation

```bash
pip install shap
```

### Example

```python
import shap
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

data=load_breast_cancer()

X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

model=RandomForestClassifier(random_state=42)
model.fit(X,y)

explainer=shap.TreeExplainer(model)
shap_values=explainer(X)

shap.plots.beeswarm(shap_values[:,:,1])
```

SHAP can provide both:

* 🌍 Global explanations
* 🎯 Local explanations

### Local Explanation

```python
sample=X.iloc[[0]]

explainer=shap.TreeExplainer(model)
shap_values=explainer(sample)

shap.plots.waterfall(shap_values[0,:,1])
```

This helps understand how individual features contributed to a particular prediction.

---

## 💡 3. LIME

**LIME (Local Interpretable Model-agnostic Explanations)** explains individual predictions by creating a simple interpretable model around a particular data point.

LIME generates slightly modified versions of an input, obtains predictions from the original model, and then learns a simple local model to approximate the original model's behavior.

### Installation

```bash
pip install lime
```

### Example

```python
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data=load_breast_cancer()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

explainer=LimeTabularExplainer(
    X_train,
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode="classification"
)

explanation=explainer.explain_instance(
    X_test[0],
    model.predict_proba
)

print(explanation.as_list())
```

LIME primarily focuses on **local explanations** for individual predictions.

---

## ⚖️ Feature Importance vs SHAP vs LIME

| Technique             | Main Purpose                     | Explanation Type |
| --------------------- | -------------------------------- | ---------------- |
| 📊 Feature Importance | Identify important features      | Global           |
| 🔍 SHAP               | Explain feature contributions    | Global + Local   |
| 💡 LIME               | Explain an individual prediction | Local            |

---

## 🔥 SHAP vs LIME

| Feature            | SHAP                           | LIME                                            |
| ------------------ | ------------------------------ | ----------------------------------------------- |
| Full Form          | SHapley Additive exPlanations  | Local Interpretable Model-agnostic Explanations |
| Global Explanation | ✅                              | ❌                                               |
| Local Explanation  | ✅                              | ✅                                               |
| Model Agnostic     | Depends on explainer           | ✅                                               |
| Main Concept       | Shapley values                 | Local surrogate model                           |
| Main Strength      | Detailed feature contributions | Simple local explanations                       |

---

## 🚨 Important Limitation

Explainability does not automatically mean causality.

If a feature has a high SHAP value, it means the feature strongly influenced the model's prediction. It does **not** prove that the feature caused the outcome.

XAI explanations should therefore be interpreted carefully and validated against domain knowledge.

---

## 🔑 Key Points

* 🧠 **XAI** helps understand machine learning predictions.
* 📊 **Feature Importance** identifies important features globally.
* 🔍 **SHAP** explains how features contribute to predictions.
* 💡 **LIME** explains individual predictions using a local surrogate model.
* 🌍 SHAP can provide both global and local explanations.
* 🎯 LIME is primarily focused on local explanations.
* ⚠️ Model explanations do not necessarily represent causal relationships.

## 🚀 XAI Workflow

```text
Dataset
   ↓
🤖 Train ML Model
   ↓
🔍 Apply XAI Technique
   ├── 📊 Feature Importance
   ├── 🔍 SHAP
   └── 💡 LIME
   ↓
📈 Analyze Feature Contributions
   ↓
🧠 Understand Model Predictions
   ↓
🤝 Build More Trustworthy ML Systems
```
