from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)
importance=pd.Series(model.feature_importances_,index=X.columns)
print(importance.sort_values(ascending=False).head(10))
importance.sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()