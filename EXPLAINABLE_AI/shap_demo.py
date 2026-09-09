import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:,:,0], X_test, feature_names=feature_names, show=False)
import matplotlib.pyplot as plt
plt.savefig("shap_summary_class0.png", dpi=300, bbox_inches='tight')