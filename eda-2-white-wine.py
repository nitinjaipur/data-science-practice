# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Shape
shape = df.shape

# head and tail
head = df.head()
tail = df.tail()

# Duplicates count
duplicates_count = df.duplicated().sum()

# Droping duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Columns
columns = df.columns.tolist()

# Counting null values
null_counts = df.isnull().sum()

# Describe
describe = df.describe()

# Value count of quality column
quality_unique = df['quality'].unique()
quality_value_counts = df['quality'].value_counts()
quality_percentile = df['quality'].value_counts(normalize=True)

# Plotting on bar chart
df['quality'].value_counts().plot(kind='bar')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.show()

# Corelation
corr = df.corr()

# Plot an heatmap for it
plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True)

# Histplot
sns.histplot(df['quality'], kde=True)
for col in columns:
    sns.histplot(df[col], kde=True)

# univariate, bivariate and multivariate
sns.pairplot(df)
