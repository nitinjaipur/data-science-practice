# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import statsmodels.api as sm

# Creating data
height = [
    160, 165, 170, 175, 180, 185, 190, 155, 168, 173, 162, 178, 174, 167, 181, 
    163, 172, 169, 177, 186, 182, 164, 179, 176, 165, 174, 168, 167, 180, 173, 
    171, 169, 178, 185, 188, 163, 174, 180, 167, 179, 175, 168, 171, 176, 161, 
    187, 170, 164
]

weight = [
    55, 57, 59, 62, 65, 68, 71, 53, 60, 63, 56, 66, 64, 59, 70, 58, 61, 60, 
    67, 73, 71, 55, 69, 65, 58, 63, 61, 60, 68, 64, 62, 61, 66, 72, 74, 56, 
    64, 70, 59, 69, 65, 60, 62, 67, 54, 75, 63, 55
]

# Creating Dataframe from the data
df = pd.DataFrame({'Height': height, 'Weight': weight})
del height
del weight

# Creating 2D data for x (features must be in 2D format, e.g. Dataframe or 2D array)
x = df[['Height']]

# Creating 1D data for y (label must be in 1D format, e.g. Series or 1D array)
y = df['Weight']

# Making scatter plot
plt.scatter(x, y)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.plot()

# Making pairplot
sns.pairplot(df)

# Splitting data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
del x
del y

# Standard scalling features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Appling LinearRegression to create the model
lr = LinearRegression()
lr.fit(x_train, y_train)

coef = lr.coef_
intercept = lr.intercept_

# Visualizing best fit line with training data
plt.scatter(x_train, y_train)
plt.plot(x_train, lr.predict(x_train))
plt.xlabel('Height')
plt.ylabel('Weight')
plt.plot()

# Making predictions
y_pred = lr.predict(x_test)

# Calculating errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_score = lr.score(x_train, y_train)
test_score = lr.score(x_test, y_test)


# OLS model
# Describe model
ols = sm.OLS(y_train, x_train)
# Fit model
ols = ols.fit()
# Summary
ols.summary()
# Prediction
y_pred_ols = ols.predict(x_test)

