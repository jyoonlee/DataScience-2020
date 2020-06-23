import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# read dataset
data = pd.read_csv('C:/Users/82105/Documents/GitHub/DataScience/TermProject/car_data.csv', encoding='utf-8')

# copy
df = data.copy()

# drop unnecessary data
df = df.drop(['New_Price'], 1)
df = df.drop(['Unnamed: 0'], 1)
columns = list(df.columns)

# preprocessing process
# change value 0 to NaN
df.replace(0, np.nan, inplace=True)
df.replace('null ', np.nan, inplace=True)

# fill value using bfill
df.fillna(method='bfill', inplace=True)
label = LabelEncoder()

# lebeling categorical value
df['Name'] = label.fit_transform(df['Name'].values)
#df['Location'] = label.fit_transform(df['Location'].values)
#df['Owner_Type'] = label.fit_transform(df['Owner_Type'].values)
#df['Fuel_Type'] = label.fit_transform(df['Fuel_Type'].values)
#df['Transmission'] = label.fit_transform(df['Transmission'].values)

loc = sns.countplot(df['Location'])
loc.set_xticklabels(loc.get_xticklabels(), rotation=40, ha="right")
plt.show()

own = sns.countplot(df['Owner_Type'])
own.set_xticklabels(own.get_xticklabels(), rotation=40, ha="right")
plt.show()

fuel = sns.countplot(df['Fuel_Type'])
fuel.set_xticklabels(fuel.get_xticklabels(), rotation=40, ha="right")
plt.show()

trans = sns.countplot(df['Transmission'])
trans.set_xticklabels(trans.get_xticklabels(), rotation=40, ha="right")
plt.show()