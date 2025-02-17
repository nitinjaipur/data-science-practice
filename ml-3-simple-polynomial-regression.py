# Importing liabraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# Generate data
X = 6 * np.random.rand(100,1) - 3
y = (0.5 * X**2) + (1.5*X) + 2 + np.random.rand(100,1)
df = pd.DataFrame({ 'Feature': X.reshape(100), 'Label': y.reshape(100) })

# Deleting unwanted data
del X, y

# Visualizing the new data
plt.scatter(df['Feature'], df['Label'], color='blue', label='Generated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Polynomial Regression')
plt.legend()
plt.show()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['Feature'], df['Label'], test_size=0.2, random_state=42)

# Convering features into polynomial features of degree 2
pf = PolynomialFeatures(degree=2, include_bias=True)
X_train = pf.fit_transform(np.array(X_train).reshape(-1, 1))
X_test = pf.transform(np.array(X_test).reshape(-1, 1))

# Creating model
lr = LinearRegression()

# Fitting data
lr.fit(X_train, y_train)

# Score
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)

# Predictions
y_pred = lr.predict(X_test)

# Calculating errors
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

# Plotting graph
X = pf.transform(np.array(df['Feature']).reshape(-1, 1))
plt.scatter(df['Feature'], df['Label'], color='g')
plt.scatter(df['Feature'], lr.predict(X), color='r')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.plot()
