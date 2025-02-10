# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Label Encoding
###############################################################################
# Creating dataframe
df = pd.DataFrame({
        'color': ['red', 'blue', 'blue', 'green', 'red', 'green']
    })

# Converting color column to numeric form using OHE
encoder = LabelEncoder()
encoded_value = encoder.fit_transform(df[['color']])

# Adding this array as column to the dataframe
df['color_encoded'] = encoded_value

# Deleting data
del df
del encoded_value
del encoder
###############################################################################

# Ordinal Encoding

# Creating dataframe
df = pd.DataFrame({
        'size': ['small', 'medium', 'medium', 'large', 'small', 'large']
    })

# Converting color column to numeric form using OHE
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
encoded_value = encoder.fit_transform(df[['size']])

# Adding this array as column to the dataframe
df['size_encoded'] = encoded_value

# Deleting data
del df
del encoded_value
del encoder
###############################################################################
