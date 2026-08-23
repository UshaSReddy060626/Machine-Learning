import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
data = load_breast_cancer()
X = data.data
y = data.target
model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42)
train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0])
train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, validation_mean, label="Validation Score")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.show()