# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data
df = pd.read_csv('winequality-red.csv', sep=';')

# Columns
columns = (df.columns).tolist()

# Shape
shape = df.shape

# Head and Tail
head = df.head()
tail = df.tail()

# Info
df.info()

# Describe
desc = df.describe()

# quality is label
# Unique values in quality
quality = df['quality'].unique()

# Value counts in quality column
quality_value_count = df['quality'].value_counts()
quality_percentile = df['quality'].value_counts(normalize=True)

# Plotting on bar chart
df['quality'].value_counts().plot(kind='bar')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.show()

# Checking null values
nulls = df.isnull().sum()

# Checking duplicates
duplicates_count = df.duplicated().sum()

# Drop duplicates
df.drop_duplicates(inplace=True)

# Corelation
corelation = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corelation, annot=True)

# Visualize the distribution
sns.histplot(df['quality'], kde=True)
for col in columns:
    sns.histplot(df[col], kde=True)

# univariate, bivariate and multivariate
sns.pairplot(df)
