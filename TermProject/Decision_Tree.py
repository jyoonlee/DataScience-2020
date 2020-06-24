import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV

df = pd.read_csv('preprocessing_data.csv', encoding='utf-8')

df['Price_ca'] = pd.qcut(df.Price, q=3, labels=[0, 1, 2])
df.drop(['Price'], 1, inplace=True)

X = df.drop(['Price_ca'], 1)
y = df['Price_ca']

kfold = KFold(5, shuffle=True)

tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)
tree.fit(X, y)

parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
              'min_samples_leaf': [1, 2, 3]}

model = GridSearchCV(tree, parameters, cv=kfold)
model.fit(X, y)

print('best value: ', model.best_params_)
print('best score: %.2f' % model.best_score_)



