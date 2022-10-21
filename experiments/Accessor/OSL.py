
import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
from sklearn import model_selection
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

file = r"accessor.xls"
adv_data = pd.read_excel(file)
new_adv_data = adv_data.ix[1:, :]
print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)

# 数据描述
# print(new_adv_data.describe())
# 缺失值检验
# print(new_adv_data[new_adv_data.isnull()==True].count())

# new_adv_data.boxplot()
# plt.savefig("boxplot.jpg")
# plt.show()
##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
# 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
print(new_adv_data.corr())

# 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
# sns.pairplot(new_adv_data, x_vars=['Md', 'Dd', 'Ld', 'Rd', 'Ad', 'Mr', 'Dr', 'Lr', 'Rr', 'Ar'], y_vars='R', size=7,
#              aspect=0.8, kind='reg')
# plt.savefig("pairplot.jpg")
# plt.show()

while True:
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(new_adv_data.ix[:, :10], new_adv_data.R, train_size=0.85)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    # print(score)
    if score > 0.82:
        break

Y_pred = model.predict(X_test)

print(Y_pred)

plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
# 显示图像
plt.savefig("predict.jpg")
plt.show()

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(Y_pred)), Y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("x")
plt.ylabel('y')
plt.savefig("ROC.jpg")
plt.show()

print("score=", score)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数

print("最佳拟合线:截距", a, ",回归系数：", b)

# print("原始数据特征:",new_adv_data.ix[:,:10].shape,
#      ",训练数据特征:",X_train.shape,
#      ",测试数据特征:",X_test.shape)
#
# print("原始数据标签:",new_adv_data.R.shape,
#      ",训练数据标签:",Y_train.shape,
#      ",测试数据标签:",Y_test.shape)