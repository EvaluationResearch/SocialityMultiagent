import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('/experiments/Sociality/23f.csv')
x=df.preyM

# 建立画布与子图对象
fig, ax = plt.subplots(figsize=(12, 8), dpi=80)  # ax 为子图对象  fig 为画布

# 绘制抖动图
sns.stripplot(x=df.pro, y=df.preyR
              , jitter=0.25  # 重要参数 抖动的幅度
              , size=8
              , ax=ax
              , linewidth=.5,
               palette=['#91bfdb','#fc8d59','olive'],edgecolor='black'
              # , palette='tab20'  # 设置所需的调色板
              )

# 装饰图片
plt.rcParams['font.sans-serif'] = ['Simhei']  # 设置字体为黑体
plt.xlabel('Prosociality for predator and prey agent', fontsize=18)  # x轴名称
plt.ylabel('Predator reward for R', fontsize=18)  # y轴名称
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.show()
