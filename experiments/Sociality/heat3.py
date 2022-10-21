import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series
import numpy as np

df3 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/3.csv')

P3=df3.P

x3=df3.x
AA=[]
xAA=[]
AE=[]
xAE=[]
AS=[]
xAS=[]
EA=[]
xEA=[]
EE=[]
xEE=[]
ES=[]
xES=[]
SA=[]
xSA=[]
SE=[]
xSE=[]
SS=[]
xSS=[]
AA3=[]
xAA3=[]
AE3=[]
xAE3=[]
AS3=[]
xAS3=[]
EA3=[]
xEA3=[]
EE3=[]
xEE3=[]
ES3=[]
xES3=[]
SA3=[]
xSA3=[]
SE3=[]
xSE3=[]
SS3=[]
xSS3=[]



for index in P3.keys():
    if P3[index] == "AVA":
        AA3.append(index)
        xAA3.append(x3[index])
        continue
    if P3[index] == "AVE":
        AE3.append(index)
        xAE3.append(x3[index])
        continue
    if P3[index] == "AVS":
        AS3.append(index)
        xAS3.append(x3[index])
        continue
    if P3[index] == "EVA":
        EA3.append(index)
        xEA3.append(x3[index])
        continue
    if P3[index] == "EVE":
        EE3.append(index)
        xEE3.append(x3[index])
        continue
    if P3[index] == "EVS":
        ES3.append(index)
        xES3.append(x3[index])
        continue
    if P3[index] == "SVA":
        SA3.append(index)
        xSA3.append(x3[index])
        continue
    if P3[index] == "SVE":
        SE3.append(index)
        xSE3.append(x3[index])
        continue
    if P3[index] == "SVS":
        SS3.append(index)
        xSS3.append(x3[index])
        continue

R=[[(sum(xAA)+sum(xAA3))/30,(sum(xAE)+sum(xAE3))/30,(sum(xAS)+sum(xAS3))/30], [(sum(xEA)+sum(xEA3))/30,(sum(xEE)+sum(xEE3))/30,(sum(xES)+sum(xES3))/30],[(sum(xSA)+sum(xSA3))/30,(sum(xSE)+sum(xSE3))/30,(sum(xSS)+sum(xSS3))/30]]


fig = plt.figure()
plt.xticks(size=15)
plt.yticks(size=15)
# sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmin=24617)
sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=15799,vmin=1726,annot_kws={"size":15})
# sns_plot = sns.heatmap(R, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['0','0.5','1'],yticklabels=['0','0.5','1'],linewidths=3)
sns_plot.set_xlabel(r'$\alpha$ for prey', fontsize=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator', fontsize=15)
plt.show()
