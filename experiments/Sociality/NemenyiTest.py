from scipy.stats import friedmanchisquare
import pandas as pd
# import Orange
import matplotlib.pyplot as plt
def toarray(df):
    b = df.values
    n = int(b.size / 2)
    dfAx = []
    for i in range(0, n):
        dfAx.append(b[i, 0])
    return dfAx
#If you want get prey reward, replace A.csv to Aprey.csv, so as S.csv and E.csv
dfA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/A.csv')
#like this , for prey
# dfA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/Aprey.csv')
dfS = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/S.csv')

dfE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/E.csv')




S=toarray(dfS)
A= toarray(dfA)
E=toarray(dfE)

stat, p = friedmanchisquare(S, A, E)
print(stat, p)
if p > 0.05:
    print('不能拒绝原假设，样本集分布相同')
else:
    print('拒绝原假设，样本集分布可能不同')

names = ['Altruistic','Egalitarian','Selfish' ]

I=[]
AR=[]
ER=[]
SR=[]
for i in range(0,len(A)):
    I.append(A[i])
    I.append(E[i])
    I.append(S[i])
    objS = pd.Series(I)
    a=objS.rank()
    AR.append(a[0])
    ER.append(a[1])
    SR.append(a[2])
    I=[]

AR=sum(AR)/180

ER=sum(ER)/180

SR=sum(SR)/180



#算法平均排名
avranks = [AR,ER,SR]
# cd = Orange.evaluation.compute_CD(avranks, 14,alpha="0.05", test="bonferroni-dunn") #tested on 14 datasets
# print("cd=",cd)
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5, textspace=1.5, cdmethod=0)
# plt.show();





CD=0.2469
h_CD=CD/2
plt.figure(figsize=(10,6))
plt.scatter(avranks,names,s=100,c='black')
for i in range(len(names)):
    yy=[names[i],names[i]]
    xx=[avranks[i]-h_CD,avranks[i]+h_CD]
    plt.plot(xx, yy,linewidth=3.0)


plt.yticks(size=25)
plt.xticks(size=25)
# plt.xticks(range(0,4,1),labels=['0','1','2','3'],size=20)


plt.xlabel("Rank",size=25)
# plt.plot(num[0],num[link],markes[link],label = '第'+ str(plt_label) + '条线段')
plt.title("Nemenyi test for predator",size=25)

plt.savefig("title"+'.png',format='PNG',dpi=500,bbox_inches='tight', pad_inches = +0.1)

plt.show()

