# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Imporing data
#df = pd.read_csv('ridge_regression_data.csv')
df = pd.read_csv('ridge_regression_multicollinear_data.csv')

# Correlation
corr = df.corr()

# Visulizing data
plt.scatter(df['Feature1'], df['Target'], color='r')
plt.scatter(df['Feature2'], df['Target'], color='g')
plt.scatter(df['Feature3'], df['Target'], color='b')
plt.xlabel('Feature')
plt.xlabel('Target')
plt.plot()

# Splitting data in features and label
features = df.iloc[:, :-1]
label = df.iloc[:, -1]

# Splitting data into testing and training set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Scalling features
sc = StandardScaler()
X_train = sc.fit_transform(np.array(X_train))
X_test = sc.transform(np.array(X_test))

# Creating model
ridge = Ridge(alpha=1.0)

# Fitting data to model
ridge.fit(X_train, y_train)

# Score
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)

# Prediction
y_pred = ridge.predict(X_test)

# Errors
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
