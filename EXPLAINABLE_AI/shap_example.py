import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)
explainer=shap.TreeExplainer(model)
shap_values=explainer(X)
shap.plots.beeswarm(shap_values[:,:,1])
plt.show()