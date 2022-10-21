import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
# Profit=pd.read_excel(r'accessor.xls')
# X=Profit.drop(columns='R')
# y=Profit['R']

Profit2=pd.read_excel(r'Two.xls')
X2=Profit2.drop(columns='R')
y2=Profit2['R']
Profit3=pd.read_excel(r'Three.xls')
X3=Profit3.drop(columns='R')
y3=Profit3['R']
X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X2, y2, test_size=0.10,random_state=0)
X_train3, X_test3, y_train3, y_test3 = model_selection.train_test_split(X3, y3, test_size=0.10,random_state=0)
X_train=pd.concat([X_train2,X_train3])
X_test=pd.concat([X_test2,X_test3])
y_train=pd.concat([y_train2,y_train3])
y_test=pd.concat([y_test2,y_test3])

##criterion 不纯度的计算方法。"mse"表示使用均方误差；"friedman_mse"表示使用费尔德曼均方误差；“mae”表示使用绝对平均误差 criterion : This parameter determines how the impurity of a split will be measured.
# random_state输入任意数字会让模型稳定下来。加上random_state这个参数后，score就不会总是变化
# mode_cv = DecisionTreeRegressor(random_state=0, criterion="mae", splitter="best",max_depth=i)
# mode_cv = DecisionTreeRegressor(random_state=5, criterion="mae",min_samples_split=15,min_samples_leaf=9,max_leaf_nodes=26, max_depth=15)
# mode_cv = DecisionTreeRegressor(splitter='random',criterion='mse',max_depth=7,min_samples_split=4,min_samples_leaf=1,min_impurity_decrease=0.2894736842105263)
# mode_cv = mode_cv.fit(X_train, y_train)
# s = mode_cv.score(X_test, y_test)
# print(s)
# CV = cross_val_score(mode_cv, X_train, y_train, cv=100,scoring='r2')
# print(CV)
# print("Cross validation Accuracy: %0.2f (+/- %0.2f)" % (CV.mean(), CV.std() * 2))

tree_para_grid={'splitter':('best','random'),
                'criterion':('mse','mae'),
                "max_depth":[*range(1,30)],
                "min_samples_split":range(2,6),
                "min_samples_leaf":[*range(1,50,5)]
                }


grid=GridSearchCV(DecisionTreeRegressor(),param_grid=tree_para_grid,cv=5)
grid.fit(X_train, y_train)

result = grid.cv_results_
print("grid.cv mean test  score::")
print(grid.cv_results_['mean_test_score'])
#The regressor.best_score_ is the average of r2 scores on left-out test folds for the best parameter combination.
print("grid best score:")
print(grid.best_score_)
print("grid best params:")
print(grid.best_params_)
pickle.dump(grid,open("dtr.dat","wb"))
# grid = pickle.load(open("dtr.dat","rb"))
y_pred1=grid.best_estimator_.predict(X_test)
sc=r2_score(y_test, grid.best_estimator_.predict(X_test))
print("r2 core:%0.2f"%sc)
CV = cross_val_score(grid, X_train, y_train, cv=5)
print(CV)
print("Cross validation Accuracy: %0.2f " % (CV.mean()))





# #保存模型

# model = pickle.load(open("dtr.dat","rb"))

y_pred=grid.predict(X_test)

print('\n',pd.DataFrame({'Prediction':y_pred,'Real':y_test}))

x=list(range(54))
plt.scatter(x, y_pred,c="slateblue")
plt.scatter(x, y_test,c="#A9561E",label="")

plt.plot(x, y_pred,color="slateblue",label="Predict value",linewidth=2)
plt.plot(x, y_test, color="#A9561E",label="Actual value", linewidth=2)

plt.title('Decision Tree Regression')
plt.xlabel('Test data')
plt.ylabel('Reward for predator team (in thousands)')

plt.legend(loc='upper right')
plt.show()

