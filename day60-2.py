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











