# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Load data
df = pd.read_csv('ridge_regression_multicollinear_data.csv')

# Correlation
corr = df.corr()

# Plotting
plt.scatter(df['Feature1'], df['Target'], colorizer='r')
plt.scatter(df['Feature2'], df['Target'], colorizer='g')
plt.scatter(df['Feature3'], df['Target'], colorizer='b')
plt.xlabel('Features')
plt.ylabel('Label')
plt.plot()


# Splitting features and label
features = df.iloc[:, :-1]
label = df.iloc[:, -1]

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Standard scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating model
lasso = Lasso()

# Training the model
lasso.fit(X_train, y_train)

# Prediction
y_pred = lasso.predict(X_test)

# Score
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)

# Errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
