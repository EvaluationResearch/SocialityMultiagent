from scipy.stats import friedmanchisquare
import pandas as pd
# import Orange
import numpy as np
import matplotlib.pyplot as plt
def toarray(df):
    b = df.values
    n = int(b.size / 2)
    dfAx = []
    for i in range(0, n):
        dfAx.append(b[i, 0])
    return dfAx

dfAA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/AA.csv')
PAA=dfAA.P
xAA1=dfAA.x1
xAA2=dfAA.x2
xAA3=dfAA.x3
dfAE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/AE.csv')
PAE=dfAE.P
xAE1=dfAE.x1
xAE2=dfAE.x2
xAE3=dfAE.x3
dfAS = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/AS.csv')
PAS=dfAS.P
xAS1=dfAS.x1
xAS2=dfAS.x2
xAS3=dfAS.x3
dfEA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/EA.csv')
PEA=dfEA.P
xEA1=dfEA.x1
xEA2=dfEA.x2
xEA3=dfEA.x3
dfEE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/EE.csv')
PEE=dfEE.P
xEE1=dfEE.x1
xEE2=dfEE.x2
xEE3=dfEE.x3
dfES = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/ES.csv')
PES=dfES.P
xES1=dfES.x1
xES2=dfES.x2
xES3=dfES.x3
dfSA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/SA.csv')
PSA=dfSA.P
xSA1=dfSA.x1
xSA2=dfSA.x2
xSA3=dfSA.x3
dfSE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/SE.csv')
PSE=dfSE.P
xSE1=dfSE.x1
xSE2=dfSE.x2
xSE3=dfSE.x3
dfSS = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/SS.csv')
PSS=dfSS.P
xSS1=dfSS.x1
xSS2=dfSS.x2
xSS3=dfSS.x3
MAA=[]
DAA=[]
LAA=[]
RAA=[]


for K in range(0,60):
    ce = PAA[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MAA.append(xAA1[K])
        if i=="D" and j==0:
            DAA.append(xAA1[K])
        if i=="L" and j==0:
            LAA.append(xAA1[K])
        if i=="R" and j==0:
            RAA.append(xAA1[K])

        if i == "M" and j == 1:
            MAA.append(xAA2[K])
        if i == "D" and j == 1:
            DAA.append(xAA2[K])
        if i == "L" and j == 1:
            LAA.append(xAA2[K])
        if i == "R" and j == 1:
            RAA.append(xAA2[K])

        if i == "M" and j == 2:
            MAA.append(xAA3[K])
        if i == "D" and j == 2:
            DAA.append(xAA3[K])
        if i == "L" and j == 2:
            LAA.append(xAA3[K])
        if i == "R" and j == 2:
            RAA.append(xAA3[K])
        j=j+1


MAE=[]
DAE=[]
LAE=[]
RAE=[]


for K in range(0,60):
    ce = PAE[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MAE.append(xAE1[K])
        if i=="D" and j==0:
            DAE.append(xAE1[K])
        if i=="L" and j==0:
            LAE.append(xAE1[K])
        if i=="R" and j==0:
            RAE.append(xAE1[K])

        if i == "M" and j == 1:
            MAE.append(xAE2[K])
        if i == "D" and j == 1:
            DAE.append(xAE2[K])
        if i == "L" and j == 1:
            LAE.append(xAE2[K])
        if i == "R" and j == 1:
            RAE.append(xAE2[K])

        if i == "M" and j == 2:
            MAE.append(xAE3[K])
        if i == "D" and j == 2:
            DAE.append(xAE3[K])
        if i == "L" and j == 2:
            LAE.append(xAE3[K])
        if i == "R" and j == 2:
            RAE.append(xAE3[K])
        j=j+1


MAS=[]
DAS=[]
LAS=[]
RAS=[]


for K in range(0,60):
    ce = PAS[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MAS.append(xAS1[K])
        if i=="D" and j==0:
            DAS.append(xAS1[K])
        if i=="L" and j==0:
            LAS.append(xAS1[K])
        if i=="R" and j==0:
            RAS.append(xAS1[K])

        if i == "M" and j == 1:
            MAS.append(xAS2[K])
        if i == "D" and j == 1:
            DAS.append(xAS2[K])
        if i == "L" and j == 1:
            LAS.append(xAS2[K])
        if i == "R" and j == 1:
            RAS.append(xAS2[K])

        if i == "M" and j == 2:
            MAS.append(xAS3[K])
        if i == "D" and j == 2:
            DAS.append(xAS3[K])
        if i == "L" and j == 2:
            LAS.append(xAS3[K])
        if i == "R" and j == 2:
            RAS.append(xAS3[K])
        j=j+1

MEA=[]
DEA=[]
LEA=[]
REA=[]


for K in range(0,60):
    ce = PEA[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MEA.append(xEA1[K])
        if i=="D" and j==0:
            DEA.append(xEA1[K])
        if i=="L" and j==0:
            LEA.append(xEA1[K])
        if i=="R" and j==0:
            REA.append(xEA1[K])

        if i == "M" and j == 1:
            MEA.append(xEA2[K])
        if i == "D" and j == 1:
            DEA.append(xEA2[K])
        if i == "L" and j == 1:
            LEA.append(xEA2[K])
        if i == "R" and j == 1:
            REA.append(xEA2[K])

        if i == "M" and j == 2:
            MEA.append(xEA3[K])
        if i == "D" and j == 2:
            DEA.append(xEA3[K])
        if i == "L" and j == 2:
            LEA.append(xEA3[K])
        if i == "R" and j == 2:
            REA.append(xEA3[K])
        j=j+1

MEE=[]
DEE=[]
LEE=[]
REE=[]


for K in range(0,60):
    ce = PEE[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MEE.append(xEE1[K])
        if i=="D" and j==0:
            DEE.append(xEE1[K])
        if i=="L" and j==0:
            LEE.append(xEE1[K])
        if i=="R" and j==0:
            REE.append(xEE1[K])

        if i == "M" and j == 1:
            MEE.append(xEE2[K])
        if i == "D" and j == 1:
            DEE.append(xEE2[K])
        if i == "L" and j == 1:
            LEE.append(xEE2[K])
        if i == "R" and j == 1:
            REE.append(xEE2[K])

        if i == "M" and j == 2:
            MEE.append(xEE3[K])
        if i == "D" and j == 2:
            DEE.append(xEE3[K])
        if i == "L" and j == 2:
            LEE.append(xEE3[K])
        if i == "R" and j == 2:
            REE.append(xEE3[K])
        j=j+1


MES=[]
DES=[]
LES=[]
RES=[]


for K in range(0,60):
    ce = PES[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MES.append(xES1[K])
        if i=="D" and j==0:
            DES.append(xES1[K])
        if i=="L" and j==0:
            LES.append(xES1[K])
        if i=="R" and j==0:
            RES.append(xES1[K])

        if i == "M" and j == 1:
            MES.append(xES2[K])
        if i == "D" and j == 1:
            DES.append(xES2[K])
        if i == "L" and j == 1:
            LES.append(xES2[K])
        if i == "R" and j == 1:
            RES.append(xES2[K])

        if i == "M" and j == 2:
            MES.append(xES3[K])
        if i == "D" and j == 2:
            DES.append(xES3[K])
        if i == "L" and j == 2:
            LES.append(xES3[K])
        if i == "R" and j == 2:
            RES.append(xES3[K])
        j=j+1


MSA=[]
DSA=[]
LSA=[]
RSA=[]


for K in range(0,60):
    ce = PSA[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MSA.append(xSA1[K])
        if i=="D" and j==0:
            DSA.append(xSA1[K])
        if i=="L" and j==0:
            LSA.append(xSA1[K])
        if i=="R" and j==0:
            RSA.append(xSA1[K])

        if i == "M" and j == 1:
            MSA.append(xSA2[K])
        if i == "D" and j == 1:
            DSA.append(xSA2[K])
        if i == "L" and j == 1:
            LSA.append(xSA2[K])
        if i == "R" and j == 1:
            RSA.append(xSA2[K])

        if i == "M" and j == 2:
            MSA.append(xSA3[K])
        if i == "D" and j == 2:
            DSA.append(xSA3[K])
        if i == "L" and j == 2:
            LSA.append(xSA3[K])
        if i == "R" and j == 2:
            RSA.append(xSA3[K])
        j=j+1


MSE=[]
DSE=[]
LSE=[]
RSE=[]


for K in range(0,60):
    ce = PSE[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MSE.append(xSE1[K])
        if i=="D" and j==0:
            DSE.append(xSE1[K])
        if i=="L" and j==0:
            LSE.append(xSE1[K])
        if i=="R" and j==0:
            RSE.append(xSE1[K])

        if i == "M" and j == 1:
            MSE.append(xSE2[K])
        if i == "D" and j == 1:
            DSE.append(xSE2[K])
        if i == "L" and j == 1:
            LSE.append(xSE2[K])
        if i == "R" and j == 1:
            RSE.append(xSE2[K])

        if i == "M" and j == 2:
            MSE.append(xSE3[K])
        if i == "D" and j == 2:
            DSE.append(xSE3[K])
        if i == "L" and j == 2:
            LSE.append(xSE3[K])
        if i == "R" and j == 2:
            RSE.append(xSE3[K])
        j=j+1

MSS=[]
DSS=[]
LSS=[]
RSS=[]


for K in range(0,60):
    ce = PSS[K].split("(")
    j = 0

    for i in ce[0]:
        if i=="M" and j==0:
            MSS.append(xSS1[K])
        if i=="D" and j==0:
            DSS.append(xSS1[K])
        if i=="L" and j==0:
            LSS.append(xSS1[K])
        if i=="R" and j==0:
            RSS.append(xSS1[K])

        if i == "M" and j == 1:
            MSS.append(xSS2[K])
        if i == "D" and j == 1:
            DSS.append(xSS2[K])
        if i == "L" and j == 1:
            LSS.append(xSS2[K])
        if i == "R" and j == 1:
            RSS.append(xSS2[K])

        if i == "M" and j == 2:
            MSS.append(xSS3[K])
        if i == "D" and j == 2:
            DSS.append(xSS3[K])
        if i == "L" and j == 2:
            LSS.append(xSS3[K])
        if i == "R" and j == 2:
            RSS.append(xSS3[K])
        j=j+1

#CHANGE THIS
MM=np.array(MSS)
DD= np.array(DSS)
LL=np.array(LSS)
RR=np.array(RSS)

MAA=[]
DAA=[]
LAA=[]
RAA=[]

stat, p = friedmanchisquare(MM, DD, LL, RR)
print(stat, p)
if p > 0.05:
    print('不能拒绝原假设，样本集分布相同')
else:
    print('拒绝原假设，样本集分布可能不同')

names = ['MADDPG','DDPG','M3DDPG','Random' ]

I=[]
MMR=[]
DDR=[]
LLR=[]
RRR=[]
ER=[]
SR=[]
for i in range(0,len(MM)):
    I.append(MM[i])
    I.append(DD[i])
    I.append(LL[i])
    I.append(RR[i])
    objS = pd.Series(I)
    a=objS.rank()
    MMR.append(a[0])
    DDR.append(a[1])
    LLR.append(a[2])
    RRR.append(a[3])
    I=[]

AR=sum(MMR)/40

ER=sum(DDR)/40

SR=sum(LLR)/40
WR=sum(RRR)/40


#算法平均排名
avranks = [AR,ER,SR,WR]
# cd = Orange.evaluation.compute_CD(avranks, 14,alpha="0.05", test="bonferroni-dunn") #tested on 14 datasets
# print("cd=",cd)
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5, textspace=1.5, cdmethod=0)
# plt.show();




#qa=3.863
CD=1.1149
h_CD=CD/2
plt.figure(figsize=(10,6))
plt.scatter(avranks,names,s=100,c='black')
for i in range(len(names)):
    yy=[names[i],names[i]]
    xx=[avranks[i]-h_CD,avranks[i]+h_CD]
    plt.plot(xx, yy,linewidth=3.0)


plt.yticks(size=20)
plt.xticks(size=20)
# plt.xticks(range(0,4,1),labels=['0','1','2','3'],size=20)


plt.xlabel("Rank",size=20)
# plt.plot(num[0],num[link],markes[link],label = '第'+ str(plt_label) + '条线段')
plt.title("SvS Nemenyi test",size=20)

plt.savefig("title"+'.png',format='PNG',dpi=500,bbox_inches='tight', pad_inches = +0.1)

plt.show()

