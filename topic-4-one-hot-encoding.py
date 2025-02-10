# Importing libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Creating dataframe
df = pd.DataFrame({
        'color': ['red', 'blue', 'blue', 'green', 'red', 'green']
    })

# Converting color column to numeric form using OHE
encoder = OneHotEncoder()
encoded_value = encoder.fit_transform(df[['color']]).toarray()
encoded_value = pd.DataFrame(encoded_value, columns=encoder.get_feature_names_out())

# Concating both dataframes
df = pd.concat([df, encoded_value], axis=1)
