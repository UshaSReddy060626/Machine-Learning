import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
X, y = load_digits(return_X_y=True)
train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='rbf', gamma=0.001), X, y, cv=5, train_sizes=np.linspace(0.1,1.0,10))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve - SVM on Digits")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.savefig("learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()