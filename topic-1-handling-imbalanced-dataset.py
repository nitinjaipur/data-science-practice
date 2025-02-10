# Importing libraries
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Setting seed
# Running the code multiple times will always output the same random number.
np.random.seed(42)

# Creating data sizes
n_samples = 1000
class_0_ratio = 0.9
n_class_0 = int(class_0_ratio * n_samples)
n_class_1 = n_samples - n_class_0

# Creating dataframes
df_majority = pd.DataFrame({
        'feature1': np.random.normal(loc=0, scale=1, size=n_class_0),
        'feature2': np.random.normal(loc=0, scale=1, size=n_class_0),
        'label': [0] * n_class_0
    })

df_minority = pd.DataFrame({
        'feature1': np.random.normal(loc=0, scale=1, size=n_class_1),
        'feature2': np.random.normal(loc=0, scale=1, size=n_class_1),
        'label': [1] * n_class_1
    })

# Deleting variables
del n_samples
del class_0_ratio
del n_class_0
del n_class_1

###############################################################################
# Upsampling minority data
df_minority_upsampled = resample(df_minority, n_samples=len(df_majority), replace=True, random_state=42)

# Concating this with majority data
df_upsampled = pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)
###############################################################################

# Downsampling majority data
df_majority_downsampled = resample(df_majority, n_samples=len(df_minority), replace=False, random_state=42)

# Concating this with minority data
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).reset_index(drop=True)
###############################################################################