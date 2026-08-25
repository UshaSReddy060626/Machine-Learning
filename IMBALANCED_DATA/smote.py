from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter
data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("Before SMOTE:",Counter(y_train))
smote=SMOTE(random_state=42)
X_train_resampled,y_train_resampled=smote.fit_resample(X_train,y_train)
print("After SMOTE:",Counter(y_train_resampled))
model=LogisticRegression(max_iter=1000)
model.fit(X_train_resampled,y_train_resampled)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))