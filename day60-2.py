# Importing packages
import numpy as np
import pandas as pd

#######################################################################
# Assignment - 1

# Creating dataframe with 6 row and 4 columns filled with random numbers
df = pd.DataFrame(np.random.randint(1, 20, size=(6, 4)), columns=['A', 'B', 'C', 'D'])

# Setting first column as index
df.set_index('A', inplace=True)

# Deleting data
del df


# Creating dataframe with 3 row and 3 columns filled with random numbers index as 'X', 'Y', 'Z' adn columns as 'A', 'B', 'C'
df = pd.DataFrame(np.random.randint(1, 20, size=(3, 3)), index=['X', 'Y', 'Z'], columns=['A', 'B', 'C'])

# Selecting data at row 'Y' and column 'B'
data_YB = df.loc['Y', 'B']

# Deleting data
del df
del data_YB
#######################################################################
# Assignment - 2

# Creating dataframe with 5 row and 3 columns filled with random numbers
df = pd.DataFrame(np.random.randint(1, 30, size=(5, 3)), columns=['A', 'B', 'C'])

# Adding new column as product of first 2 columns
df['D'] = df['A'] * df['B']

# Deleting data
del df


# Creating dataframe with 5 row and 3 columns filled with random numbers
df = pd.DataFrame(np.random.randint(1, 30, size=(3, 4)))

# Row wise sum
sum_row = df.sum(axis=1)

# Column wise sum
sum_col = df.sum(axis=0)

# Deleting data
del df
del sum_row
del sum_col
#######################################################################
# Assignment - 3

# Creating dataframe with 5 row and 3 columns filled with random numbers
df = pd.DataFrame(np.random.randint(1, 100, size=(5, 3)))

# Replacing some values with nan
df.iloc[1, 2] = np.nan
df.iloc[0, 0] = np.nan
df.iloc[2, 1] = np.nan

# Filling nan with mean of respective column
df.fillna(df.mean(), inplace=True)

# Deleting data
del df


# Creating dataframe with 6 row and 4 columns filled with random numbers
df = pd.DataFrame(np.random.randint(1, 100, size=(6, 4)))

# Replacing some values with nan
df.iloc[0, 0] = np.nan
df.iloc[2, 1] = np.nan
df.iloc[1, 2] = np.nan
df.iloc[3, 3] = np.nan

# Filling nan with mean of respective column
df.dropna(inplace=True)

# Deleting data
del df
#######################################################################
# Assignment - 4

# Creating dataframe with column Categoty (random chices from 'A', 'B', 'C') and Valus (random numbers)
df = pd.DataFrame(
        {
         'Category': np.random.choice(['A', 'B', 'C'], size=10),
         'Value': np.random.randint(1, 100, size=10)
        }
    )

# Grouping data based on 'Category' column and finding mean of Values for each group
group = df.groupby('Category')['Value'].mean()

# Deleting data
del df
del group


# Creating dataframe with column Product (random chices from 'Pen', 'Notebook', 'Eraser') Categoty (random chices from 'A', 'B', 'C') and Valus (random numbers)
df = pd.DataFrame(
        {
            'Product': np.random.choice(['Pen', 'Notebook', 'Eraser'], size=10),
            'Category': np.random.choice(['A', 'B', 'C'], size=10),
            'Sales': np.random.randint(1, 100, size=10)
        }
    )

# Grouping data based on 'Category' column and finding mean of Sales for each group
group = df.groupby('Category')['Sales'].sum()

# Deleting data
del df
del group
#######################################################################
# Assignment - 5

# First dataframe of length 3
df1 = pd.DataFrame(
        {
            'Key': ['A', 'B', 'C'],
            'Value': np.random.randint(1, 100, size=3)
        }
    )

# First dataframe of length 4
df2 = pd.DataFrame(
        {
            'Key': ['A', 'B', 'C', 'D'],
            'Value': np.random.randint(5, 200, size=4)
        }
    )

# Merging dataframes on column 'Key'
df_merge_key = pd.merge(df1, df2, on='Key')

# Concatenating dataframes row-wise
df_concat_row = pd.concat([df1, df2], axis=0)

# Concatenating dataframes column-wise
df_concat_col = pd.concat([df1, df2], axis=1)

# Deleting data
del df1
del df2
del df_merge_key
del df_concat_row
del df_concat_col
#######################################################################
# Assignment - 6

# Creating datetime range
datetimeRange = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

# Creating dataframe
df = pd.DataFrame({
        'Date': datetimeRange,
        'Value': np.random.randint(1, 200, size=len(datetimeRange))
    })

# Setting column 'Date' as index
df.set_index('Date', inplace=True)

# Resampling data on monthly basis and finding mean
df_monthly_mean = df.resample('M').mean()

# Rolling mean of window of 7 days
df_rolling_mean = df.rolling(window=7).mean()

# Deleting data
del df
del datetimeRange
del df_monthly_mean
del df_rolling_mean
#######################################################################
# Assignment - 7

# Index array
index = [['A', 'A', 'B', 'B', 'C', 'C'], ['1', '2', '1', '2', '1', '2']]

# Creating MultiIndex from index array
index = pd.MultiIndex.from_arrays(index, names=('Category', 'Subcategory'))

# Creating dataframe
df = pd.DataFrame(np.random.randint(1, 20, size=(6, 3)), index=index, columns=['Col1', 'Col2', 'Col3'])

# Selecting from 'A' and '1'
df.loc[('A', '1'), 'Col1']

# Grouping and sum based on 'Category' and 'Subcategory'
group_sum = df.groupby(['Category', 'Subcategory']).sum()

# Deleting data
del df
del index
del group_sum
#######################################################################
# Assignment - 8

# Creating datetime range
dateTime = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')

# Creating dataframe
df = pd.DataFrame({
        'Date': np.random.choice(dateTime, size=100),
        'Category': np.random.choice(['a', 'b', 'c'], size=100),
        'Value': np.random.randint(1, 10, size=100)
    })

# Pivot table for sum of 'Value' for each 'Category' by 'Date
df_sum = df.pivot_table(values='Value', columns='Category', index='Date', aggfunc='sum')

# Deleting data
del df
del dateTime
del df_sum


# Creating dataframe
df = pd.DataFrame({
        'Year': np.random.choice(['2021', '2022', '2023', '2024'], size=100),
        'Quater': np.random.choice(['Q1', 'Q2', 'Q3' , 'Q4'], size=100),
        'Revenue': np.random.randint(1, 100, size=100)
    })

# Pivot table for mean of 'Revenue' for each 'Quater' by 'Year
df_mean_revenue = df.pivot_table(values='Revenue', columns='Quater', index='Year', aggfunc='mean')

# Deleting data
del df
del df_mean_revenue
#######################################################################
# Assignment - 9

# Creating dataframe
df = pd.DataFrame(np.random.randint(1, 20, size=(5, 3)))

# Doubling each element using applymap
df_doubled = df.applymap(lambda x: x * 2)

# Deleting data
del df
del df_doubled


# Creating dataframe
df = pd.DataFrame(np.random.randint(1, 20, size=(6, 3)))

# Adding values of all column for each row and assigning rsult to new column 'Sum'
df['Sum'] = df.apply(lambda x: x.sum(), axis=1)

# Deleting data
del df
#######################################################################
# Assignment - 10

# Creating seriese with array
mySeriese = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])

# Convering to upper case
mySeriese_upper = mySeriese.str.upper()

# Extracting first 3 char from each
mySeriese_first_three = mySeriese.str[:3]

# Deleting data
del mySeriese
del mySeriese_upper
del mySeriese_first_three
#######################################################################
