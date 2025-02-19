# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV

# Loading data
df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header=1)

# Getting info
df.info()

# Getting all null records
null_df = df[df.isnull().any(axis=1)]

# This dataset have onle 2 null records so we can drop these records
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
del null_df

# Getting stats
stats = df.describe()

# Listing all the columns
columns = df.columns.tolist()

# We can see some column names have whitespaces before and after names. We need to remove whitespaces
df.columns = df.columns.str.strip()

# Finding all non numeric rows
non_numeric_rows = df[df.applymap(lambda x: isinstance(x, str)).any(axis=1)]

# Converting dataset to numeric
# This will remove last column from data frame
df_numeric = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Adding last column 'Classes' again to dataframe
df = pd.concat([df_numeric, df['Classes']], axis=1)
del df_numeric

# So now non numeric data from numeric has became Nan so we can drop them
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# By watching the data we can say that columns 'day', 'month', 'year', 'Temperature', 'RH', 'Ws' are of int type
# So we Conert these columns to int type
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

# Finding uniques in column 'Classes'
unique_classes = df['Classes'].unique()

# Converting to 0 and 1
df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)

# year column have single value(2012) so drop it
df.drop('year', axis=1, inplace=True)

# Density plot
plt.style.use('seaborn-v0_8')
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Pie chart for classes
classes_percentage = df['Classes'].value_counts(normalize=True) * 100
classes_label = ['Fire', 'Not fire']
plt.figure(figsize=(12, 7))
plt.pie(classes_percentage,labels=classes_label, autopct='%1.1f%%')
plt.title('Pie chart for Classes')
plt.plot()

# Correlation
corr = df.corr()

# Heat map for corr
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True)

# Box plot for FWI
sns.boxplot(df['FWI'], color='green')

# Monthly fire analysis
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=df)
plt.xlabel('Month', weight='bold')
plt.ylabel('Count', weight='bold')
plt.plot()

# For making prediction we don't need day, month and Classes column
# So we are removing these columns
df.drop(['day', 'month', 'Classes'], axis=1, inplace=True)

# Handling Multicollinearity
def getMulticollinearColumns(df, threshold, exclude):
    df = df.drop(exclude, axis=1)
    corr = df.corr()
    columns = df.columns.tolist()
    multicollinearColumns = set()
    for column in columns:
        correlations = corr[column]        
        for index, correlation in correlations.items():
            if (index != column) and (column not in multicollinearColumns) and (correlation >= threshold ):
                multicollinearColumns.add(index)
    return multicollinearColumns

multicollinearColumns = getMulticollinearColumns(df, 0.85, ['FWI'])

# Drop multicollinearColumns from dataframe
df.drop(multicollinearColumns, axis=1, inplace=True)
del multicollinearColumns, corr

# Splitting data into features and label
features = df.drop('FWI', axis=1)
label = df['FWI']

# Spliting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Scalling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating and training LinearRegression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_train_score = lr.score(X_train, y_train)
lr_test_score = lr.score(X_test, y_test)

# Creating and training Lasso model
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_train_score = lasso.score(X_train, y_train)
lasso_test_score = lasso.score(X_test, y_test)

# Creating and training Ridge model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
ridge_train_score = ridge.score(X_train, y_train)
ridge_test_score = ridge.score(X_test, y_test)

# Creating and training ElasticNet model
en = ElasticNet(alpha=1.0)
en.fit(X_train, y_train)
y_pred_ridge = en.predict(X_test)
en_train_score = en.score(X_train, y_train)
en_test_score = en.score(X_test, y_test)

# Creating and training LassoCV model
lassoCV = LassoCV()
lassoCV.fit(X_train, y_train)
y_pred_ridge = lassoCV.predict(X_test)
lassoCV_train_score = lassoCV.score(X_train, y_train)
lassoCV_test_score = lassoCV.score(X_test, y_test)

# Creating and training LassoCV model
ridgeCV = RidgeCV()
ridgeCV.fit(X_train, y_train)
y_pred_ridge = ridgeCV.predict(X_test)
ridgeCV_train_score = ridgeCV.score(X_train, y_train)
ridgeCV_test_score = ridgeCV.score(X_test, y_test)

# Creating and training ElasticNet model
enCV = ElasticNetCV()
enCV.fit(X_train, y_train)
y_pred_ridge = enCV.predict(X_test)
enCV_train_score = enCV.score(X_train, y_train)
enCV_test_score = enCV.score(X_test, y_test)
