from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
train_data=list(zip(X_train,y_train))
majority=[row for row in train_data if row[1]==1]
minority=[row for row in train_data if row[1]==0]
majority_undersampled=resample(majority,replace=False,n_samples=len(minority),random_state=42)
balanced_data=majority_undersampled+minority
X_train_balanced=[row[0] for row in balanced_data]
y_train_balanced=[row[1] for row in balanced_data]
model=LogisticRegression(max_iter=1000)
model.fit(X_train_balanced,y_train_balanced)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))