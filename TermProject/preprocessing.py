import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read dataset
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/82105/Desktop/train-data.csv', encoding='utf-8')

print(df.isnull().sum())
del df['New_Price']
columns = list(df.columns)

print(columns)
label = LabelEncoder()

#preprocessing
df = df.applymap(str)
df['Name'] = label.fit_transform(df['Name'].values)
print(np.unique(df['Location'])) # dirty value detection
df['Location'] = label.fit_transform(df['Location'].values)
print(np.unique(df['Location']))
df['Year'] = pd.to_numeric(df['Year'], errors="coerce")
df['Year'].fillna(method='bfill', inplace=True)
print(np.unique(df['Year']))
df['Kilometers_Driven'] = pd.to_numeric(df['Kilometers_Driven'], errors="coerce")
df['Kilometers_Driven'].fillna(method='bfill', inplace=True)
df['Kilometers_Driven']=df['Kilometers_Driven'].where(df['Kilometers_Driven'].between(0,300000))
print(np.unique(df['Fuel_Type']))
df['Fuel_Type'].replace({'?': np.NaN, 'nan': np.NaN}, inplace=True)
df['Fuel_Type'].fillna(method='bfill', inplace=True)
df['Fuel_Type'] = label.fit_transform(df['Fuel_Type'].values)
print(np.unique(df['Fuel_Type']))
print(np.unique(df['Transmission']))
df['Transmission'].replace({'?': np.NaN, 'nan': np.NaN}, inplace=True)
df['Transmission'].fillna(method='bfill', inplace=True)
df['Transmission'] = label.fit_transform(df['Transmission'].values)
print(np.unique(df['Transmission']))
print(np.unique(df['Owner_Type']))
df['Owner_Type'].replace({'?': np.NaN, 'nan': np.NaN}, inplace=True)
df['Owner_Type'].fillna(method='bfill', inplace=True)
df['Owner_Type'] = label.fit_transform(df['Owner_Type'].values)
print(np.unique(df['Owner_Type']))
df['Mileage (kmpl)'] = pd.to_numeric(df['Mileage (kmpl)'], errors="coerce")
df['Mileage (kmpl)'].replace({0: np.NaN}, inplace=True)
df['Mileage (kmpl)'].fillna(method='bfill', inplace=True)
df['Engine (CC)'] = pd.to_numeric(df['Engine (CC)'], errors="coerce")
df['Engine (CC)'].fillna(method='bfill', inplace=True)
df['Engine (CC)'].replace({0: np.NaN}, inplace=True)
df['Power (bhp)'] = pd.to_numeric(df['Power (bhp)'], errors="coerce")
df['Power (bhp)'].fillna(method='bfill', inplace=True)
df['Seats'] = pd.to_numeric(df['Seats'], errors="coerce")
df['Seats'].replace({0: np.NaN}, inplace=True)
df['Seats'].fillna(method='bfill', inplace=True)
df['Price'] = pd.to_numeric(df['Price'], errors="coerce")
df['Price'].fillna(method='bfill', inplace=True)

for each in columns:
    print(df[each].value_counts())

#heat map
heatmap_data = df[['Year', 'Kilometers_Driven', 'Mileage (kmpl)', 'Engine (CC)', 'Power (bhp)', 'Seats', 'Price']]
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 16})
#plt.show()
#print(heatmap_data)
