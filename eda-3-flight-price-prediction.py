# Import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Loading data
df = pd.read_excel('flight_price.xlsx')

# Info
df.info()

# Describe
describe = df.describe()

# Columns
columns = (df.columns).tolist()

# We alraedy have Source and Destination as columns so we can drop Route column
df.drop('Route', axis=1, inplace=True)

# We will make 3 new columns from Date_of_Journey
# Journey_Date, Journey_Month, Journey_Year
df['Journey_Date'] = df['Date_of_Journey'].str.split('/').str[0]
df['Journey_Month'] = df['Date_of_Journey'].str.split('/').str[1]
df['Journey_Year'] = df['Date_of_Journey'].str.split('/').str[2]

# Converting datatype from object to int
df['Journey_Date'] = df['Journey_Date'].astype(int)
df['Journey_Month'] = df['Journey_Month'].astype(int)
df['Journey_Year'] = df['Journey_Year'].astype(int)

# Droping Date_of_Journey column
df.drop('Date_of_Journey', axis=1, inplace=True)

# We will make 2 new columns from Dep_Time
# Dep_Hour, Dep_Minute
df['Dep_Hour'] = df['Dep_Time'].str.split(':').str[0]
df['Dep_Minute'] = df['Dep_Time'].str.split(':').str[1]

# Converting datatype from object to int
df['Dep_Hour'] = df['Dep_Hour'].astype(int)
df['Dep_Minute'] = df['Dep_Minute'].astype(int)

# Droping Dep_Time column
df.drop('Dep_Time', axis=1, inplace=True)

# Removing extra unwanted data from values of Arrival_Time column
df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: x.split(' ')[0])

# We will make 2 new columns from Arrival_Time
# Arrival_Hour, Arrival_Minute
df['Arrival_Hour'] = df['Arrival_Time'].str.split(':').str[0]
df['Arrival_Minute'] = df['Arrival_Time'].str.split(':').str[1]

# Converting datatype from object to int
df['Arrival_Hour'] = df['Arrival_Hour'].astype(int)
df['Arrival_Minute'] = df['Arrival_Minute'].astype(int)

# Droping Arrival_Time column
df.drop('Arrival_Time', axis=1, inplace=True)

# Function to convert hour-minute string to minutes
def convertStringToMinutes(string):
    string = string.split(' ')
    minutes = 0
    for e in string:
      if e[-1] == 'h':
        minutes += int(e[:-1]) * 60
      elif e[-1] == 'm':
        minutes += int(e[:-1])
    return minutes

# Converting Duration column values using convertStringToMinutes
df['Duration'] = df['Duration'].apply(lambda x: convertStringToMinutes(x))

# Total_Stops column have ordinal data
# So we will doing ordinal encoding for this column

# Dropping empty Total_Stops data
df.dropna(subset=['Total_Stops'], inplace=True)

# Listing all unique values of Total_Stops column
unique_Total_Stops = df['Total_Stops'].unique().tolist()

# Creating dictionary to map Total_Stops column
Total_Stops_dict = {
        "non-stop" : 0,
        "1 stop" : 1,
        "2 stops" : 2,
        "3 stops" : 3,
        "4 stops" : 4,
    }

# Ordinal encoding Total_Stops column using Total_Stops_dict
df['Total_Stops'] = df['Total_Stops'].map(Total_Stops_dict)

# Encoding Source column using OneHotEncoder
encoder = OneHotEncoder()
Source_encoder = encoder.fit_transform(df[['Source']]).toarray()
Source_encoder = pd.DataFrame(Source_encoder, columns=encoder.get_feature_names_out())

# Deleting Source column from df
df.drop('Source', axis=1, inplace=True)

# Concating Source_encoder with df
df = pd.concat([df, Source_encoder], axis=1)

# Deleting encoder and Source_encoder
del encoder
del Source_encoder

# Encoding Destination column using OneHotEncoder
encoder = OneHotEncoder()
Destination_encoder = encoder.fit_transform(df[['Destination']]).toarray()
Destination_encoder = pd.DataFrame(Destination_encoder, columns=encoder.get_feature_names_out())

# Deleting Destination column from df
df.drop('Destination', axis=1, inplace=True)

# Concating Source_encoder with df
df = pd.concat([df, Destination_encoder], axis=1)

# Deleting encoder and Source_encoder
del encoder
del Destination_encoder

# Encoding Airline column using OneHotEncoder
encoder = OneHotEncoder()
Airline_encoder = encoder.fit_transform(df[['Airline']]).toarray()
Airline_encoder = pd.DataFrame(Airline_encoder, columns=encoder.get_feature_names_out())

# Deleting Airline column from df
df.drop('Airline', axis=1, inplace=True)

# Concating Source_encoder with df
df = pd.concat([df, Airline_encoder], axis=1)

# Deleting encoder and Source_encoder
del encoder
del Airline_encoder

# Counting null values in Additional_Info column
null_count_Additional_Info = df['Additional_Info'].isnull().sum()

# Dropping empty Additional_Info data
df.dropna(subset=['Additional_Info'], inplace=True)

# Listing all unique values of Additional_Info column
unique_Additional_Info = df['Additional_Info'].unique().tolist()

# Encoding Additional_Info column using OneHotEncoder
encoder = OneHotEncoder()
Additional_Info_encoder = encoder.fit_transform(df[['Additional_Info']]).toarray()
Additional_Info_encoder = pd.DataFrame(Additional_Info_encoder, columns=encoder.get_feature_names_out())

# Deleting Airline column from df
df.drop('Additional_Info', axis=1, inplace=True)

# Concating Source_encoder with df
df = pd.concat([df, Additional_Info_encoder], axis=1)

# Deleting encoder and Source_encoder
del encoder
del Additional_Info_encoder

# Finding unique in Journey_Year
year_unique = df['Journey_Year'].unique()

# There is just 1 value in it so we can remove this column
df.drop('Journey_Year', axis=1, inplace=True)
