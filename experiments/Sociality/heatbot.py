import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series
import numpy as np
df2 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/2.csv')
df3 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/3.csv')
P2=df2.P
P3=df3.P
x=df2.x
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

R=[[(sum(xAA)+sum(xAA3))/40,(sum(xAE)+sum(xAE3))/40,(sum(xAS)+sum(xAS3))/40], [(sum(xEA)+sum(xEA3))/40,(sum(xEE)+sum(xEE3))/40,(sum(xES)+sum(xES3))/40],[(sum(xSA)+sum(xSA3))/40,(sum(xSE)+sum(xSE3))/40,(sum(xSS)+sum(xSS3))/40]]


fig = plt.figure()
plt.xticks(size=15)
plt.yticks(size=15)
# sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmin=24617)
sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=15799,vmin=1726,annot_kws={"size":15})
# sns_plot = sns.heatmap(R, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['0','0.5','1'],yticklabels=['0','0.5','1'],linewidths=3)
sns_plot.set_xlabel(r'$\alpha$ for prey', fontsize=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator', fontsize=15)
plt.show()

plt.xticks(size=15)
label = 'AvA', 'AvE', 'AvS', 'EvA', 'EvE', 'EvS', 'SvA', 'SvE', 'SvS'
plt.xlabel('Sociality for predator and prey', fontsize=15)  # x轴名称
plt.ylabel('Predator reward(in thousands)', fontsize=15)  # y轴名称
xAA.extend(xAA3)
xAAn=np.array(xAA)
AAf=Series(xAAn)

xAE.extend(xAE3)
xAEn=np.array(xAE)
AEf=Series(xAEn)

xAS.extend(xAS3)
xASn=np.array(xAS)
ASf=Series(xASn)

xEA.extend(xEA3)
xEAn=np.array(xEA)
EAf=Series(xEAn)

xEE.extend(xEE3)
xEEn=np.array(xEE)
EEf=Series(xEEn)


xES.extend(xES3)
xESn=np.array(xES)
ESf=Series(xESn)

xSA.extend(xSA3)
xSAn=np.array(xSA)
SAf=Series(xSAn)

xSE.extend(xSE3)
xSEn=np.array(xSE)
SEf=Series(xSEn)

xSS.extend(xSS3)
xSSn=np.array(xSS)
SSf=Series(xSSn)

plt.boxplot([AAf, AEf, ASf,EAf,EEf, ESf, SAf, SEf,SSf], labels=label)
# plt.boxplot([xAA.append(xAA3), xAE.append(xAE3), xAS.append(xAS3),xEA.append(xEA3), xEE.append(xEE3), xES.append(xES3), xSA.append(xSA3), xSE.append(xSE3),xSS.append(xSS3)], labels=label)
plt.show()