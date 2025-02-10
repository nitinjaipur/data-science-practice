# Importing libraries
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Generating data
features, label = make_classification(
        n_samples=1000, n_features=2,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        weights=[0.90], random_state=42
    )

# Creating dataframes
features = pd.DataFrame(features, columns=['f1', 'f2'])
label = pd.DataFrame(label, columns=['target'])

# Concating dataframes
df = pd.concat([features, label], axis=1)

# Deleting data
del features
del label

# Plotting data before upsampling
plt.scatter(df['f1'], df['f2'], c=df['target'])


# Upsampling using smote
smote = SMOTE()
features, label = smote.fit_resample(df[['f1', 'f2']], df['target'])

# Deleting data
del smote

# Concating dataframes
df_upsampled = pd.concat([features, label], axis=1)

# Deleting data
del features
del label

# Plotting data after upsampling
plt.scatter(df_upsampled['f1'], df_upsampled['f2'], c=df_upsampled['target'])

