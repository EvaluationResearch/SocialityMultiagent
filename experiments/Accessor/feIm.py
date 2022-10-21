import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection
Profit=pd.read_excel(r'accessor.xls')
Profit=pd.read_excel(r'accessor.xls')
X=Profit.drop(columns='R')
y=Profit['R']
MdA=[]
DdA=[]
LdA=[]
RdA=[]
AdA=[]
MrA=[]
DrA=[]
LrA=[]
RrA=[]
ArA=[]

table = xlrd.open_workbook("accessor.xls")
sheet = table.sheet_by_name("Sheet2")
row_count = sheet.nrows
column_count = sheet.ncols
for i in range(1, row_count):
    for j in range(0, column_count):
        cell = sheet.cell(i, j)
        if j == 0:
            Md=cell.value
        if j==1:
            Dd=cell.value
        if j==2:
            Ld=cell.value
        if j == 3:
            Rd = cell.value
        if j == 4:
            Ad = cell.value
        if j == 5:
            Mr = cell.value
        if j == 6:
            Dr = cell.value
        if j == 7:
            Lr = cell.value
        if j == 8:
            Rr = cell.value
        if j == 9:
            Ar = cell.value
        if j==10:
            Re=cell.value
    Av=Re/(Md+Dd+Ld+Rd+Ad+Mr+Dr+Lr+Rr+Ar)
    MdA.append(Av*Md)
    DdA.append(Av*Dd)
    LdA.append(Av*Ld)
    RdA.append(Av*Rd)
    AdA.append(Av*Ad)
    MrA.append(Av*Mr)
    DrA.append(Av*Dr)
    LrA.append(Av*Lr)
    RrA.append(Av*Rr)
    ArA.append(Av*Ar)
print(sum(MdA))
print(sum(DdA))
print(sum(LdA))
print(sum(RdA))
print(sum(AdA))
print(sum(MrA))
print(sum(DrA))
print(sum(LrA))
print(sum(RrA))
print(sum(ArA))
x1=('Md','Dd','Ld','Rd','Ad','Mr','Dr','Lr','Rr','Ar')
y1=(int(sum(MdA)),int(sum(DdA)),int(sum(LdA)),int(sum(RdA)),int(sum(AdA)),int(sum(MrA)),int(sum(DrA)),int(sum(LrA)),int(sum(RrA)),int(sum(ArA)))
plt.bar(x1, y1)
for xx, yy in zip(x1,y1):
    plt.text(xx, yy, str(yy), ha='center')

plt.title("Feature Importances",fontsize=12)
plt.xlabel("Feature Names",fontsize=12)
plt.show()
plt.plot(x1, y1, color="#A9561E",label="Actual value", linewidth=2)

plt.title('Decision Tree Regression')
plt.xlabel('Test data')
plt.ylabel('Reward for predator team (in thousands)')

print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.10)
# train, test=model_selection.train_test_split(Profit, test_size=0.05, random_state=1)
# model=LogisticRegression()
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)
# fig,ax=plt.subplot(figsize=(10,10))
print(model.feature_importances_)
feature_names=['Md','Dd','Ld','Rd','Ad','Mr','Dr','Lr','Rr','Ar']
plt.barh(range(len(feature_names)),model.feature_importances_)
plt.title("Feature Importances",fontsize=12)
plt.ylabel("Feature Names",fontsize=12)
plt.yticks(range(10),feature_names)
plt.show()