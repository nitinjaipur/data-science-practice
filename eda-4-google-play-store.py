# Importing libraries
import pandas as pd
from numpy import nan

# Loading data
df = pd.read_csv('googleplaystore.csv')

# Info
df.info()

# Columns
columns = df.columns.tolist()

# Type and Genres column is irrelavant so remove it
df.drop(['Type', 'Genres'], axis=1, inplace=True)

# Find skewness of Rating
rating_skewness = df['Rating'].dropna().skew()
# skewness is less than 1 so it is normally distributed
# So we will fill nan with mean
mean_rating = round(df['Rating'].mean(), 1)
df['Rating'].fillna(mean_rating, inplace=True)


# Checking if all data in Reviews column are numeric
review_non_numreic_count = (~df['Reviews'].str.isnumeric()).sum()
# Making new dataframe for these record
review_non_numreic = df[~df['Reviews'].str.isnumeric()]
# Converting M in Reviews column to multiply by number accordingly
review_non_numreic['Reviews'] = review_non_numreic['Reviews'].apply(lambda x : int(float(x.split('M')[0]) * 1000000))
# Removing these records from main df
df.drop(review_non_numreic.index, inplace=True)
# Concating both dataframes
df = pd.concat([df, review_non_numreic])
# Conerting Reviews column from object to int
df['Reviews'] = df['Reviews'].astype(int)



# Creating a function to clean Size column
def convertSize(size):
    size = size.replace(',', '').replace('+', '')
    if size[-1] == 'M':
        size = int(float(size.split('M')[0]) * 100000)
    elif size[-1] == 'k':
        size = int(float(size.split('k')[0]) * 1000)
    elif size.isnumeric():        
        size = int(float(size))
    else:
        size = nan
    return size

# Finding unique values of Size column
unique_size = df['Size'].unique()
# Applying convertSize function to Size column to clean data
df['Size'] = df['Size'].apply(convertSize)
# Finding skewness of Size column
skewness_size = df['Size'].dropna().skew()
# Skewness is more than 1 so we will impute nan values with median
df['Size'] = df['Size'].fillna(df['Size'].median())
# Converting Size column to int type
df['Size'] = df['Size'].astype(int)




# Creating a function to clean Installs column
def convertInstalls(install):
    install = install.replace(',', '').replace('+', '')
    if install.isnumeric():        
        install = int(float(install))
    else:
        install = nan
    return install

# Finding unique values of Size column
unique_Installs = df['Installs'].unique()
# Applying convertSize function to Size column to clean data
df['Installs'] = df['Installs'].apply(convertInstalls)
# Finding skewness of Size column
skewness_Installs = df['Installs'].dropna().skew()
# Skewness is more than 1 so we will impute nan values with median
df['Installs'] = df['Installs'].fillna(df['Installs'].median())
# Converting Size column to int type
df['Installs'] = df['Installs'].astype(int)


# Finding unique values of Size column
unique_Price = df['Price'].unique()

# Creating a function to clean Installs column
def convertPrice(price):
    price = price.replace('$', '')
    try:
      price = float(price)
    except:
      price = 'nan'
    return price

# Applying convertPrice function to Price column to clean data
df['Price'] = df['Price'].apply(convertInstalls)
# Finding skewness of Price column
skewness_Price = df['Price'].dropna().skew()
# Skewness is 0 so we will impute nan values with mean
df['Price'] = df['Price'].fillna(df['Price'].mean())





