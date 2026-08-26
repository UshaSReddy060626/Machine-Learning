from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)
explainer=LimeTabularExplainer(
    X.values,
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode="classification")
explanation=explainer.explain_instance(
    X.iloc[0].values,
    model.predict_proba)
print(explanation.as_list())