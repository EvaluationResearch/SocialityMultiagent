import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series
import numpy as np
df2 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/2.csv')

P2=df2.P

x=df2.x

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
for index in P2.keys():
    if P2[index] == "AVA":
        AA.append(index)
        xAA.append(x[index])
        continue
    if P2[index] == "AVE":
        AE.append(index)
        xAE.append(x[index])
        continue
    if P2[index] == "AVS":
        AS.append(index)
        xAS.append(x[index])
        continue
    if P2[index] == "EVA":
        EA.append(index)
        xEA.append(x[index])
        continue
    if P2[index] == "EVE":
        EE.append(index)
        xEE.append(x[index])
        continue
    if P2[index] == "EVS":
        ES.append(index)
        xES.append(x[index])
        continue
    if P2[index] == "SVA":
        SA.append(index)
        xSA.append(x[index])
        continue
    if P2[index] == "SVE":
        SE.append(index)
        xSE.append(x[index])
        continue
    if P2[index] == "SVS":
        SS.append(index)
        xSS.append(x[index])
        continue



R=[[(sum(xAA)+sum(xAA3))/10,(sum(xAE)+sum(xAE3))/10,(sum(xAS)+sum(xAS3))/10], [(sum(xEA)+sum(xEA3))/10,(sum(xEE)+sum(xEE3))/10,(sum(xES)+sum(xES3))/10],[(sum(xSA)+sum(xSA3))/10,(sum(xSE)+sum(xSE3))/10,(sum(xSS)+sum(xSS3))/10]]


fig = plt.figure()
plt.xticks(size=15)
plt.yticks(size=15)
# sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmin=24617)
sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=15799,vmin=1726,annot_kws={"size":15})
# sns_plot = sns.heatmap(R, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['0','0.5','1'],yticklabels=['0','0.5','1'],linewidths=3)
sns_plot.set_xlabel(r'$\alpha$ for prey', fontsize=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator', fontsize=15)
plt.show()
