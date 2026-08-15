import numpy as np
from sklearn.linear_model import LinearRegression

# Area in square feet
X = np.array([[1000],[1500],[2000],[2500],[3000]])

# Price in lakhs
y = np.array([30, 45, 60, 75, 90])
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[2200]])
print("Predicted Price:", prediction[0], "lakhs")