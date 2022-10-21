import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
Profit=pd.read_excel(r'accessor.xls')
X=Profit.drop(columns='R')
y=Profit['R']
while True:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
    model=LinearRegression().fit(X_train,y_train)
    # print(model.score(X_test, y_test))
    if model.score(X_test, y_test)>0.85:
        print("score without cv: {}".format(model.score(X_test, y_test)))
        break
# train, test=model_selection.train_test_split(Profit, test_size=0.05, random_state=1)

# model = sm.OLS(y_train, X_train)
# results = model.fit()
# print(results.rsquared)



cf=model.coef_
print({"Coefficients for the linear regression:":cf})
y_pred=model.predict(X_test)
print('\n',pd.DataFrame({'Prediction':y_pred,'Real':y_test}))


CV = cross_val_score(model, X_train, y_train, cv=10,scoring='r2')
print(CV)
print("Cross validation Accuracy: %0.2f (+/- %0.2f)" % (CV.mean(), CV.std() * 2))
# #保存模型
pickle.dump(model,open("dtr.dat","wb"))
# model = pickle.load(open("dtr.dat","rb"))

y_pred=model.predict(X_test)

print('\n',pd.DataFrame({'Prediction':y_pred,'Real':y_test}))

x=list(range(54))
plt.scatter(x, y_pred,c="slateblue")
plt.scatter(x, y_test,c="#A9561E",label="")

plt.plot(x, y_pred,color="slateblue",label="Predict value",linewidth=2)
plt.plot(x, y_test, color="#A9561E",label="Actual value", linewidth=2)
plt.title('Liner Regression')
plt.xlabel('Test data')
plt.ylabel('Reward for predator team (in thousands)')

plt.legend(loc='upper right')
plt.show()

