import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('preprocessing_data.csv', encoding='utf-8')

df['Price_ca'] = pd.qcut(df.Price, q=3, labels=[0, 1, 2])

# price value
X = df.drop(['Price_ca'], 1)
X = df.drop(['Price'], 1)
y = df['Price']

# k-fold
parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              'min_samples_leaf': [1, 2, 3, 4]}

kfold = KFold(5, shuffle=True)
tree2 = DecisionTreeRegressor(max_depth=5, random_state=0)
model = GridSearchCV(tree2, parameters, cv=kfold)
model.fit(X, y)

print('best value: ', model.best_params_)
print('best score: %.2f' % model.best_score_)

# categorical value
df.drop(['Price'], 1, inplace=True)

X = df.drop(['Price_ca'], 1)
y = df['Price_ca']

print('==========================================================\n')

# test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)

# confusion matrix
confusion = confusion_matrix(y_test, prediction)
sns.heatmap(pd.DataFrame(confusion), annot=True, fmt='g')
plt.title('Decision Tree')
plt.ylabel('Real value')
plt.xlabel('Prediction value')

print(classification_report(y_test, prediction, digits=3, zero_division=1))
plt.show()

print('==========================================================\n')

# k-fold
parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              'min_samples_leaf': [1, 2, 3, 4]}

kfold = KFold(5, shuffle=True)
model = GridSearchCV(tree, parameters, cv=kfold)
model.fit(X, y)

print('best value: ', model.best_params_)
print('best score: %.2f' % model.best_score_)
