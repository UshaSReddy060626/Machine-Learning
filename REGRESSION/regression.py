import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

#create dataset

#hours studied

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])

#exam scores

y = np.array([35,42,50,58,65,72,78,85])


#split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print("Training data:")
print(X_train)
print("\nTesting data:")
print(X_test)


#create linear regression model

model = LinearRegression()


#train the model

model.fit(X_train, y_train)


#get model parameters
print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("\nEquation:")
print(f"y = {model.intercept_:.2f} + "f"{model.coef_[0]:.2f}x")


#make predictions

y_pred = model.predict(X_test)
print("\nActual values:")
print(y_test)
print("\nPredicted values:")
print(y_pred)


#calculate errors

errors = y_test - y_pred
print("\nErrors:")
print(errors)



#calculate evaluation metrics

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nEvaluation Metrics")
print("------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)



#predict a new student's score

new_student = np.array([[9]])
prediction = model.predict(new_student)
print("\nPrediction for 9 hours of study:")
print(f"{prediction[0]:.2f} marks")


#visualize regression line

plt.scatter(X, y)
plt.plot(X,model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression")
plt.show()