# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Reading data
df = pd.read_csv('economic_index.csv')

# Listing column names
columns = df.columns.tolist()

# Dropping unwanted columns
df.drop(['Unnamed: 0', 'year', 'month'], axis=1, inplace=True)

# Correlation
corr = df.corr()

# Finding null values
null_count = df.isnull().sum()

# Visulizing relationship between features and label
# interest_rate and index_price
plt.scatter(df['interest_rate'], df['index_price'])
plt.xlabel('Interest Rate')
plt.ylabel('Index Price')
plt.show()

# unemployment_rate and index_price
plt.scatter(df['unemployment_rate'], df['index_price'])
plt.xlabel('Unemployment Rate')
plt.ylabel('Index Price')
plt.show()


# Regrassion plots
# interest_rate and index_price
sns.regplot(x='interest_rate', y='index_price', data=df)
# unemployment_rate and index_price
sns.regplot(x='unemployment_rate', y='index_price', data=df)

# deleting unnecessary data
del columns, corr, null_count

# Separating features and label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# deleting unnecessary data
del X,y

# Scalling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting data to model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Parametrs
coef = lr.coef_
intercept = lr.intercept_

# Score
training_score = lr.score(X_train, y_train)
testing_score = lr.score(X_test, y_test)

# Predictions
y_pred = lr.predict(X_test)

# Errors
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# OLS
ols = sm.OLS(y_train, X_train)
ols = ols.fit()
y_pred_ols = ols.predict(X_test)
ols.summary()

