# Importing libraries
import numpy as np
import seaborn as sns

# Creating an array
arr = [124, 154, 1, 10, 130, 145, 166, 153, 415, 166, 176]

# 5 numbers of summary
minimum, Q1, average, Q3, maximum = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1])

# Finding IQR
IQR = Q3 - Q1

# Findinf lower and upper limit for detect outliers
lower_fence = Q1 - (1.5 * IQR)
upper_fence = Q3 + (1.5 * IQR)

# Plotiing box plot
sns.boxplot(arr)
