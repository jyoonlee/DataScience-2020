import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV

df = pd.read_csv('preprocessing_data.csv', encoding='utf-8')

# Split
X = df.drop(['Price'], 1)
y = df['Price']

kfold = KFold(5, shuffle=True)
parameters = {'n_neighbors': range(3, 10)}
clf = GridSearchCV(KNeighborsRegressor(),
                   parameters, cv=kfold)
clf.fit(X, y)

print('best k-value: ', clf.best_params_)
print('best score: %.2f' % clf.best_score_)
