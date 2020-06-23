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
print(df.isnull().sum())

# drop unnecessary data
df = df.drop(['New_Price'], 1)
df = df.drop(['Unnamed: 0'], 1)
columns = list(df.columns)
print(columns)

# preprocessing process
# change value 0 to NaN
df.replace(0, np.nan, inplace=True)
df.replace('null ', np.nan, inplace=True)

# fill value using bfill
df.fillna(method='bfill', inplace=True)
label = LabelEncoder()

# lebeling categorical value
df['Name'] = label.fit_transform(df['Name'].values)
df['Location'] = label.fit_transform(df['Location'].values)
df['Owner_Type'] = label.fit_transform(df['Owner_Type'].values)
df['Fuel_Type'] = label.fit_transform(df['Fuel_Type'].values)
df['Transmission'] = label.fit_transform(df['Transmission'].values)

print(df.info())
df.to_csv("preprocessing_data.csv")

# heat map with non-categorical value
heatmap_data = df[['Year', 'Kilometers_Driven', 'Mileage (kmpl)', 'Engine (CC)', 'Power (bhp)', 'Seats', 'Price']]
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 16})

#plt.show()
print(heatmap_data)

df.drop(['Kilometers_Driven', 'Mileage (kmpl)', 'Seats'], 1)
print(df)

# Min-Max Scaling
X = df[['Year', 'Engine (CC)', 'Power (bhp)']]
y = df['Price']
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df,columns=['Year', 'Engine (CC)', 'Power (bhp)'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['Year'], ax=ax1)
sns.kdeplot(df['Engine (CC)'], ax=ax1)
sns.kdeplot(df['Power (bhp)'], ax=ax1)

ax2.set_title('After Robust Scaling')
sns.kdeplot(scaled_df['Year'], ax=ax2)
sns.kdeplot(scaled_df['Engine (CC)'], ax=ax2)
sns.kdeplot(scaled_df['Power (bhp)'], ax=ax2)
plt.show()

