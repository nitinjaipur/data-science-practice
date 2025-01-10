# Importing packages
import numpy as np

#######################################################################
# Assignment - 1

# Creating array of size (5, 5) with random numbers between 1 and 20 
arr = np.random.randint(1, 21, size=(5, 5))

# Replacing all values of 3rd column with 1
arr[:, 2] = 1

# Delete arr
del arr


# Creating array of size (4, 4) with random numbers between 1 and 16 
arr = np.random.randint(1, 17, size=(4, 4))

# Replacing all values diagonal with 0
np.fill_diagonal(arr, 0)

# Delete arr
del arr
#######################################################################

# Assignment - 2

# Creating array of size (6, 6) with random numbers between 1 and 36
arr = np.random.randint(1, 37, size=(6, 6))

# Extracting row 3 to 5 and columns 2 to 4
sub_arr = arr[2:6, 1:5]

# Delete arr
del arr
del sub_arr


# Creating array of size (5, 5) with random numbers between 1 and 20 
arr = np.random.randint(1, 21, size=(5, 5))

# Extracting first row
arr1 = arr[0]

# Extracting 2nd to last 2nd row and 1st and last number
# Reshaping to 1 direction array
arr2 = arr[1:-1, [0, -1]].ravel()

# Extracting first row
arr3 = arr[arr.shape[0] - 1]

# Merging all array
sub_arr = np.concatenate((arr1, arr2, arr3))

# Delete arr
del arr
del sub_arr
del arr1
del arr2
del arr3
#######################################################################

# Assignment - 3

# Creatingn 2 arrays of size (3, 4) with random numbers between 1 and 20
arr1 = np.random.randint(1, 21, size=(3, 4))
arr2 = np.random.randint(1, 21, size=(3, 4))

# Adding arrays
arr_sum = np.add(arr1, arr2)

# Substracting arrays
arr_sub = np.subtract(arr1, arr2)

# Multipling arrays
arr_mul = np.multiply(arr1, arr2)

# Divide arrays
arr_div = np.divide(arr1, arr2)

# Delete arr
del arr1
del arr2
del arr_sum
del arr_sub
del arr_mul
del arr_div


# Creating array of size (4, 4) from numbers from 1 to 16
arr = np.arange(1, 17).reshape(4, 4)

# Column wise sum
sum_col = np.sum(arr, axis=0)

# Row wise sum
sum_row = np.sum(arr, axis=1)

# Delete arr
del arr
del sum_col
del sum_row
#######################################################################

# Assignment - 4

# Creating array of size (5, 5) with random numbers between 1 and 16
arr = np.random.randint(1, 16, size=(5, 5))

# Mean
mean = np.mean(arr)

# Median
median = np.median(arr)

# Standard Daviation
std = np.std(arr)

# Variation
var = np.var(arr)

# Deleting data
del arr
del mean
del median
del std
del var


# Creating array of size (3, 3) with random numbers between 1 and 10
arr = np.random.randint(1, 10, size=(3, 3))

# Mean
mean = np.mean(arr)

# Standard Daviation
std = np.std(arr)

# Normalalization of array
arr_normalized = np.divide(np.subtract(arr, mean), std)

# Deleting data
del arr
del mean
del std
del arr_normalized
#######################################################################

# Assignment - 5

# Creating array of size (3, 3) with random numbers between 1 and 10
arr_2d = np.random.randint(1, 10, size=(3, 3))

# Creating array of size (3) with random numbers between 1 and 10
arr_1d = np.random.randint(1, 10, size=(3))

# Adding arr_1d to each row of arr_2d
arr_result = np.add(arr_2d, arr_1d)

# Deleting data
del arr_2d
del arr_1d
del arr_result


# Creating array of size (4, 4) with random numbers between 1 and 10
arr_2d = np.random.randint(1, 10, size=(4, 4))

# Creating array of size (3,) with random numbers between 1 and 10
arr_1d = np.random.randint(1, 10, size=(4))

# Substracting arr_1d from each row of arr_2d
arr_result = np.subtract(arr_2d, arr_1d)

# Deleting data
del arr_2d
del arr_1d
del arr_result
#######################################################################

# Assignment - 6

# Creating array of size (3, 3) with random numbers between 1 and 10
arr = np.random.randint(1, 10, size=(3, 3))

# Determinant
determinant = np.linalg.det(arr)

# Inverse
inverse = np.linalg.inv(arr)

# Eigen Values
eigenvalues = np.linalg.eigvals(arr)

# Deleting data
del arr
del determinant
del inverse
del eigenvalues


# Creating array of size (2, 3) with random numbers between 1 and 10
arr1 = np.random.randint(1, 10, size=(2, 3))

# Creating array of size (3, 2) with random numbers between 1 and 10
arr2 = np.random.randint(1, 10, size=(3, 2))

# Matrix multiplication
arr_matrix_mult = np.dot(arr1, arr2)

# Deleting data
del arr1
del arr2
del arr_matrix_mult
#######################################################################

# Assignment - 7

# Creating array of size (3, 3) with random numbers between 1 and 10
arr = np.random.randint(1, 10, size=(3, 3))

# Reshaping to size (1, 9)
arr = arr.reshape((1, 9))

# Reshaping to size (9, 1)
arr= arr.reshape((9, 1))

# Deleting data
del arr


# Creating array of size (5, 5) with random numbers between 1 and 10
arr = np.random.randint(1, 10, size=(5, 5))

# Reshaping to 1 direction
arr = arr.ravel()

# Reshaping to size (5, 5)
arr = arr.reshape((5, 5))

# Deleting data
del arr
#######################################################################

# Assignment - 8

# Creating array of size (5, 5) with random numbers between 1 and 10
arr = np.random.randint(1, 10, size=(5, 5))

# Selecting corner elements using fancy indexing
arr_corner = arr[[0, 0, -1, -1], [0, -1, 0, -1]]

# Deleting data
del arr
del arr_corner

# Creating array of size (5, 5) with random numbers between 1 and 20
arr = np.random.randint(1, 20, size=(4, 4))

# Using boolean index i.e. filters to convert values greater than 10 to 10
arr[arr > 10] = 10

# Deleting data
del arr
#######################################################################

# Assignment - 9

# Defining data types
data_types = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]

# Creating structered array
arr = np.array([('Neil', 27, 65),('Ben', 22, 72.2), ('Kevin', 38, 82.2)], dtype=data_types)

# Sorting array based on age
arr_shorted = np.sort(arr, order='age')

# Deleting data
del arr
del arr_shorted
del data_types


# Defining data types
data_types = [('point 1', 'i4'), ('point 2', 'i4')]

# Creating structered array
arr = np.array([(27, 65),(22, 72), (38, 82)], dtype=data_types)

# Sorting array based on age
arr_distances = np.sqrt((arr['point 1'][:, np.newaxis] - arr['point 1'])**2 + (arr['point 2'][:, np.newaxis] - arr['point 2'])**2)

# Deleting data
del arr
del data_types
del arr_distances
#######################################################################