import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series
import numpy as np
df2 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/2.csv')
df3 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/3.csv')
Re=df2.PdVPr
Re3=df3.PdVPr

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
x1=df2.x1
x2=df2.x2

x11=df3.x1
x22=df3.x2
x33=df3.x3
MAA=[]
DAA=[]
LAA=[]
RAA=[]
for k in AA:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAA.append(x1[k])
        if i=="D" and j==0:
            DAA.append(x1[k])
        if i=="L" and j==0:
            LAA.append(x1[k])
        if i=="R" and j==0:
            RAA.append(x1[k])

        if i == "M" and j == 1:
            MAA.append(x2[k])
        if i == "D" and j == 1:
            DAA.append(x2[k])
        if i == "L" and j == 1:
            LAA.append(x2[k])
        if i == "R" and j == 1:
            RAA.append(x2[k])
        j = j + 1

for k in AA3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAA.append(x11[k])
        if i=="D" and j==0:
            DAA.append(x11[k])
        if i=="L" and j==0:
            LAA.append(x11[k])
        if i=="R" and j==0:
            RAA.append(x11[k])

        if i == "M" and j == 1:
            MAA.append(x22[k])
        if i == "D" and j == 1:
            DAA.append(x22[k])
        if i == "L" and j == 1:
            LAA.append(x22[k])
        if i == "R" and j == 1:
            RAA.append(x22[k])

        if i == "M" and j == 2:
            MAA.append(x33[k])
        if i == "D" and j == 2:
            DAA.append(x33[k])
        if i == "L" and j == 2:
            LAA.append(x33[k])
        if i == "R" and j == 2:
            RAA.append(x33[k])
        j = j + 1

MAE=[]
DAE=[]
LAE=[]
RAE=[]
for k in AE:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAE.append(x1[k])
        if i=="D" and j==0:
            DAE.append(x1[k])
        if i=="L" and j==0:
            LAE.append(x1[k])
        if i=="R" and j==0:
            RAE.append(x1[k])

        if i == "M" and j == 1:
            MAE.append(x2[k])
        if i == "D" and j == 1:
            DAE.append(x2[k])
        if i == "L" and j == 1:
            LAE.append(x2[k])
        if i == "R" and j == 1:
            RAE.append(x2[k])
        j = j + 1

for k in AE3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAE.append(x11[k])
        if i=="D" and j==0:
            DAE.append(x11[k])
        if i=="L" and j==0:
            LAE.append(x11[k])
        if i=="R" and j==0:
            RAE.append(x11[k])

        if i == "M" and j == 1:
            MAE.append(x22[k])
        if i == "D" and j == 1:
            DAE.append(x22[k])
        if i == "L" and j == 1:
            LAE.append(x22[k])
        if i == "R" and j == 1:
            RAE.append(x22[k])

        if i == "M" and j == 2:
            MAE.append(x33[k])
        if i == "D" and j == 2:
            DAE.append(x33[k])
        if i == "L" and j == 2:
            LAE.append(x33[k])
        if i == "R" and j == 2:
            RAE.append(x33[k])
        j = j + 1

MAS=[]
DAS=[]
LAS=[]
RAS=[]
for k in AS:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAS.append(x1[k])
        if i=="D" and j==0:
            DAS.append(x1[k])
        if i=="L" and j==0:
            LAS.append(x1[k])
        if i=="R" and j==0:
            RAS.append(x1[k])

        if i == "M" and j == 1:
            MAS.append(x2[k])
        if i == "D" and j == 1:
            DAS.append(x2[k])
        if i == "L" and j == 1:
            LAS.append(x2[k])
        if i == "R" and j == 1:
            RAS.append(x2[k])
        j = j + 1
for k in AS3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MAS.append(x11[k])
        if i=="D" and j==0:
            DAS.append(x11[k])
        if i=="L" and j==0:
            LAS.append(x11[k])
        if i=="R" and j==0:
            RAS.append(x11[k])

        if i == "M" and j == 1:
            MAS.append(x22[k])
        if i == "D" and j == 1:
            DAS.append(x22[k])
        if i == "L" and j == 1:
            LAS.append(x22[k])
        if i == "R" and j == 1:
            RAS.append(x22[k])

        if i == "M" and j == 2:
            MAS.append(x33[k])
        if i == "D" and j == 2:
            DAS.append(x33[k])
        if i == "L" and j == 2:
            LAS.append(x33[k])
        if i == "R" and j == 2:
            RAS.append(x33[k])
        j = j + 1
MEA=[]
DEA=[]
LEA=[]
REA=[]
for k in EA:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MEA.append(x1[k])
        if i=="D" and j==0:
            DEA.append(x1[k])
        if i=="L" and j==0:
            LEA.append(x1[k])
        if i=="R" and j==0:
            REA.append(x1[k])

        if i == "M" and j == 1:
            MEA.append(x2[k])
        if i == "D" and j == 1:
            DEA.append(x2[k])
        if i == "L" and j == 1:
            LEA.append(x2[k])
        if i == "R" and j == 1:
            REA.append(x2[k])
        j = j + 1
for k in EA3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MEA.append(x11[k])
        if i=="D" and j==0:
            DEA.append(x11[k])
        if i=="L" and j==0:
            LEA.append(x11[k])
        if i=="R" and j==0:
            REA.append(x11[k])

        if i == "M" and j == 1:
            MEA.append(x22[k])
        if i == "D" and j == 1:
            DEA.append(x22[k])
        if i == "L" and j == 1:
            LEA.append(x22[k])
        if i == "R" and j == 1:
            REA.append(x22[k])

        if i == "M" and j == 2:
            MEA.append(x33[k])
        if i == "D" and j == 2:
            DEA.append(x33[k])
        if i == "L" and j == 2:
            LEA.append(x33[k])
        if i == "R" and j == 2:
            REA.append(x33[k])
        j = j + 1
MEE=[]
DEE=[]
LEE=[]
REE=[]
for k in EE:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MEE.append(x1[k])
        if i=="D" and j==0:
            DEE.append(x1[k])
        if i=="L" and j==0:
            LEE.append(x1[k])
        if i=="R" and j==0:
            REE.append(x1[k])

        if i == "M" and j == 1:
            MEE.append(x2[k])
        if i == "D" and j == 1:
            DEE.append(x2[k])
        if i == "L" and j == 1:
            LEE.append(x2[k])
        if i == "R" and j == 1:
            REE.append(x2[k])
        j = j + 1
for k in EE3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MEE.append(x11[k])
        if i=="D" and j==0:
            DEE.append(x11[k])
        if i=="L" and j==0:
            LEE.append(x11[k])
        if i=="R" and j==0:
            REE.append(x11[k])

        if i == "M" and j == 1:
            MEE.append(x22[k])
        if i == "D" and j == 1:
            DEE.append(x22[k])
        if i == "L" and j == 1:
            LEE.append(x22[k])
        if i == "R" and j == 1:
            REE.append(x22[k])

        if i == "M" and j == 2:
            MEE.append(x33[k])
        if i == "D" and j == 2:
            DEE.append(x33[k])
        if i == "L" and j == 2:
            LEE.append(x33[k])
        if i == "R" and j == 2:
            REE.append(x33[k])
        j = j + 1
MES=[]
DES=[]
LES=[]
RES=[]
for k in ES:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MES.append(x1[k])
        if i=="D" and j==0:
            DES.append(x1[k])
        if i=="L" and j==0:
            LES.append(x1[k])
        if i=="R" and j==0:
            RES.append(x1[k])

        if i == "M" and j == 1:
            MES.append(x2[k])
        if i == "D" and j == 1:
            DES.append(x2[k])
        if i == "L" and j == 1:
            LES.append(x2[k])
        if i == "R" and j == 1:
            RES.append(x2[k])
        j = j + 1
for k in ES3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MES.append(x11[k])
        if i=="D" and j==0:
            DES.append(x11[k])
        if i=="L" and j==0:
            LES.append(x11[k])
        if i=="R" and j==0:
            RES.append(x11[k])

        if i == "M" and j == 1:
            MES.append(x22[k])
        if i == "D" and j == 1:
            DES.append(x22[k])
        if i == "L" and j == 1:
            LES.append(x22[k])
        if i == "R" and j == 1:
            RES.append(x22[k])

        if i == "M" and j == 2:
            MES.append(x33[k])
        if i == "D" and j == 2:
            DES.append(x33[k])
        if i == "L" and j == 2:
            LES.append(x33[k])
        if i == "R" and j == 2:
            RES.append(x33[k])
        j = j + 1
MSA=[]
DSA=[]
LSA=[]
RSA=[]
for k in SA:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSA.append(x1[k])
        if i=="D" and j==0:
            DSA.append(x1[k])
        if i=="L" and j==0:
            LSA.append(x1[k])
        if i=="R" and j==0:
            RSA.append(x1[k])

        if i == "M" and j == 1:
            MSA.append(x2[k])
        if i == "D" and j == 1:
            DSA.append(x2[k])
        if i == "L" and j == 1:
            LSA.append(x2[k])
        if i == "R" and j == 1:
            RSA.append(x2[k])
        j = j + 1
for k in SA3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSA.append(x11[k])
        if i=="D" and j==0:
            DSA.append(x11[k])
        if i=="L" and j==0:
            LSA.append(x11[k])
        if i=="R" and j==0:
            RSA.append(x11[k])

        if i == "M" and j == 1:
            MSA.append(x22[k])
        if i == "D" and j == 1:
            DSA.append(x22[k])
        if i == "L" and j == 1:
            LSA.append(x22[k])
        if i == "R" and j == 1:
            RSA.append(x22[k])

        if i == "M" and j == 2:
            MSA.append(x33[k])
        if i == "D" and j == 2:
            DSA.append(x33[k])
        if i == "L" and j == 2:
            LSA.append(x33[k])
        if i == "R" and j == 2:
            RSA.append(x33[k])
        j = j + 1
MSE=[]
DSE=[]
LSE=[]
RSE=[]
for k in SE:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSE.append(x1[k])
        if i=="D" and j==0:
            DSE.append(x1[k])
        if i=="L" and j==0:
            LSE.append(x1[k])
        if i=="R" and j==0:
            RSE.append(x1[k])

        if i == "M" and j == 1:
            MSE.append(x2[k])
        if i == "D" and j == 1:
            DSE.append(x2[k])
        if i == "L" and j == 1:
            LSE.append(x2[k])
        if i == "R" and j == 1:
            RSE.append(x2[k])
        j = j + 1
for k in SE3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSE.append(x11[k])
        if i=="D" and j==0:
            DSE.append(x11[k])
        if i=="L" and j==0:
            LSE.append(x11[k])
        if i=="R" and j==0:
            RSE.append(x11[k])

        if i == "M" and j == 1:
            MSE.append(x22[k])
        if i == "D" and j == 1:
            DSE.append(x22[k])
        if i == "L" and j == 1:
            LSE.append(x22[k])
        if i == "R" and j == 1:
            RSE.append(x22[k])

        if i == "M" and j == 2:
            MSE.append(x33[k])
        if i == "D" and j == 2:
            DSE.append(x33[k])
        if i == "L" and j == 2:
            LSE.append(x33[k])
        if i == "R" and j == 2:
            RSE.append(x33[k])
        j = j + 1
MSS=[]
DSS=[]
LSS=[]
RSS=[]
for k in SS:
    ce=Re[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSS.append(x1[k])
        if i=="D" and j==0:
            DSS.append(x1[k])
        if i=="L" and j==0:
            LSS.append(x1[k])
        if i=="R" and j==0:
            RSS.append(x1[k])

        if i == "M" and j == 1:
            MSS.append(x2[k])
        if i == "D" and j == 1:
            DSS.append(x2[k])
        if i == "L" and j == 1:
            LSS.append(x2[k])
        if i == "R" and j == 1:
            RSS.append(x2[k])
        j = j + 1
for k in SS3:
    ce=Re3[k].split("(")
    j = 0
    for i in ce[0]:

        if i=="M" and j==0:
            MSS.append(x11[k])
        if i=="D" and j==0:
            DSS.append(x11[k])
        if i=="L" and j==0:
            LSS.append(x11[k])
        if i=="R" and j==0:
            RSS.append(x11[k])

        if i == "M" and j == 1:
            MSS.append(x22[k])
        if i == "D" and j == 1:
            DSS.append(x22[k])
        if i == "L" and j == 1:
            LSS.append(x22[k])
        if i == "R" and j == 1:
            RSS.append(x22[k])

        if i == "M" and j == 2:
            MSS.append(x33[k])
        if i == "D" and j == 2:
            DSS.append(x33[k])
        if i == "L" and j == 2:
            LSS.append(x33[k])
        if i == "R" and j == 2:
            RSS.append(x33[k])
        j = j + 1


RM=[[sum(MAA)/40,sum(MAE)/40,sum(MAS)/40], [sum(MEA)/40,sum(MEE)/40,sum(MES)/40],[sum(MSA)/40,sum(MSE)/40,sum(MSS)/40]]
RD=[[sum(DAA)/40,sum(DAE)/40,sum(DAS)/40], [sum(DEA)/40,sum(DEE)/40,sum(DES)/40],[sum(DSA)/40,sum(DSE)/40,sum(DSS)/40]]
RL=[[sum(LAA)/40,sum(LAE)/40,sum(LAS)/40], [sum(LEA)/40,sum(LEE)/40,sum(LES)/40],[sum(LSA)/40,sum(LSE)/40,sum(LSS)/40]]
RR=[[sum(RAA)/40,sum(RAE)/40,sum(RAS)/40], [sum(REA)/40,sum(REE)/40,sum(RES)/40],[sum(RSA)/40,sum(RSE)/40,sum(RSS)/40]]


# sns_plot = sns.heatmap(R, annot=True, cmap='OrRd',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmin=24617)
plt.xticks(size=15)
plt.yticks(size=15)
# sns_plot = sns.heatmap(RM, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=211645,vmin=16943)
sns_plot = sns.heatmap(RM, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=5200,vmin=400,annot_kws={"size":15})
# sns_plot = sns.heatmap(R, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['0','0.5','1'],yticklabels=['0','0.5','1'],linewidths=3)
sns_plot.set_xlabel(r'$\alpha$ for prey with MADDPG',size=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator with MADDPG',size=15)
plt.show()
plt.xticks(size=15)
plt.yticks(size=15)
sns_plot = sns.heatmap(RD, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=5200,vmin=400,annot_kws={"size":15})
sns_plot.set_xlabel(r'$\alpha$ for prey with DDPG',size=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator with DDPG',size=15)
plt.show()
plt.xticks(size=15)
plt.yticks(size=15)
sns_plot = sns.heatmap(RL, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=5200,vmin=400,annot_kws={"size":15})
sns_plot.set_xlabel(r'$\alpha$ for prey with M3DDPG',size=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator with M3DDPG',size=15)
plt.show()
plt.xticks(size=15)
plt.yticks(size=15)
sns_plot = sns.heatmap(RR, annot=True, cmap='YlGnBu',fmt=".0f",xticklabels=['A','E','S'],yticklabels=['A','E','S'],linewidths=3,vmax=5200,vmin=400,annot_kws={"size":15})
sns_plot.set_xlabel(r'$\alpha$ for prey with random',size=15)  # x轴标题
sns_plot.set_ylabel(r'$\alpha$ for predator with random',size=15)
plt.show()
xAAn=np.array(MAA)
MAAf=Series(xAAn)
xAAn=np.array(DAA)
DAAf=Series(xAAn)
xAAn=np.array(LAA)
LAAf=Series(xAAn)
xAAn=np.array(RAA)
RAAf=Series(xAAn)

xAEn=np.array(MAE)
MAEf=Series(xAEn)
xAEn=np.array(DAE)
DAEf=Series(xAEn)
xAEn=np.array(LAE)
LAEf=Series(xAEn)
xAEn=np.array(RAE)
RAEf=Series(xAEn)

xASn=np.array(MAS)
MASf=Series(xASn)
xASn=np.array(DAS)
DASf=Series(xASn)
xASn=np.array(LAS)
LASf=Series(xASn)
xASn=np.array(RAS)
RASf=Series(xASn)

xEAn=np.array(MEA)
MEAf=Series(xEAn)
xEAn=np.array(DEA)
DEAf=Series(xEAn)
xEAn=np.array(LEA)
LEAf=Series(xEAn)
xEAn=np.array(REA)
REAf=Series(xEAn)

xEEn=np.array(MEE)
MEEf=Series(xEEn)
xEEn=np.array(DEE)
DEEf=Series(xEEn)
xEEn=np.array(LEE)
LEEf=Series(xEEn)
xEEn=np.array(REE)
REEf=Series(xEEn)

xESn=np.array(MES)
MESf=Series(xESn)
xESn=np.array(DES)
DESf=Series(xESn)
xESn=np.array(LES)
LESf=Series(xESn)
xESn=np.array(RES)
RESf=Series(xESn)

xSAn=np.array(MSA)
MSAf=Series(xSAn)
xSAn=np.array(DSA)
DSAf=Series(xSAn)
xSAn=np.array(LSA)
LSAf=Series(xSAn)
xSAn=np.array(RSA)
RSAf=Series(xSAn)

xSEn=np.array(MSE)
MSEf=Series(xSEn)
xSEn=np.array(DSE)
DSEf=Series(xSEn)
xSEn=np.array(LSE)
LSEf=Series(xSEn)
xSEn=np.array(RSE)
RSEf=Series(xSEn)


xSSn=np.array(MSS)
MSSf=Series(xSSn)
xSSn=np.array(DSS)
DSSf=Series(xSSn)
xSSn=np.array(RSS)
RSSf=Series(xSSn)
xSSn=np.array(LSS)
LSSf=Series(xSSn)


label = 'AvA', 'AvE', 'AvS', 'EvA', 'EvE', 'EvS', 'SvA', 'SvE', 'SvS'
plt.xticks(size=15)
plt.xlabel('Sociality for predator and prey', fontsize=17)  # x轴名称
plt.ylabel('Predator reward for MADDPG(in thousands)', fontsize=15)  # y轴名称


plt.boxplot([MAAf, MAEf, MASf,MEAf,MEEf, MESf, MSAf, MSEf,MSSf], labels=label)
# plt.boxplot([xAA.append(xAA3), xAE.append(xAE3), xAS.append(xAS3),xEA.append(xEA3), xEE.append(xEE3), xES.append(xES3), xSA.append(xSA3), xSE.append(xSE3),xSS.append(xSS3)], labels=label)
plt.show()

label = 'AvA', 'AvE', 'AvS', 'EvA', 'EvE', 'EvS', 'SvA', 'SvE', 'SvS'
plt.xticks(size=15)
plt.xlabel('Sociality for predator and prey', fontsize=17)  # x轴名称
plt.ylabel('Predator reward for DDPG(in thousands)', fontsize=15)  # y轴名称


plt.boxplot([DAAf, DAEf, DASf,DEAf,DEEf, DESf, DSAf, DSEf,DSSf], labels=label)
# plt.boxplot([xAA.append(xAA3), xAE.append(xAE3), xAS.append(xAS3),xEA.append(xEA3), xEE.append(xEE3), xES.append(xES3), xSA.append(xSA3), xSE.append(xSE3),xSS.append(xSS3)], labels=label)
plt.show()
label = 'AvA', 'AvE', 'AvS', 'EvA', 'EvE', 'EvS', 'SvA', 'SvE', 'SvS'
plt.xticks(size=15)
plt.xlabel('Sociality for predator and prey', fontsize=17)  # x轴名称
plt.ylabel('Predator reward for M3DDPG(in thousands)', fontsize=15)  # y轴名称


plt.boxplot([LAAf, LAEf, LASf,LEAf,LEEf, LESf, LSAf, LSEf,LSSf], labels=label)
# plt.boxplot([xAA.append(xAA3), xAE.append(xAE3), xAS.append(xAS3),xEA.append(xEA3), xEE.append(xEE3), xES.append(xES3), xSA.append(xSA3), xSE.append(xSE3),xSS.append(xSS3)], labels=label)
plt.show()
label = 'AvA', 'AvE', 'AvS', 'EvA', 'EvE', 'EvS', 'SvA', 'SvE', 'SvS'
plt.xticks(size=15)
plt.xlabel('Sociality for predator and prey', fontsize=17)  # x轴名称
plt.ylabel('Predator reward for random(in thousands)', fontsize=15)  # y轴名称


plt.boxplot([RAAf, RAEf, RASf,REAf,REEf, RESf, RSAf, RSEf,RSSf], labels=label)
# plt.boxplot([xAA.append(xAA3), xAE.append(xAE3), xAS.append(xAS3),xEA.append(xEA3), xEE.append(xEE3), xES.append(xES3), xSA.append(xSA3), xSE.append(xSE3),xSS.append(xSS3)], labels=label)
plt.show()