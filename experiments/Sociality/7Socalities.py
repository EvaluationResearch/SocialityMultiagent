import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series
import numpy as np
df2 = pd.read_csv('/experiments/Accessor/7Soci.csv')
# df3 = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/3.csv')
P2=df2.PdVPr
x=df2.x
y=df2.y

xpred0=[]
xpred01=[]
xpred03=[]
xpred05=[]
xpred07=[]
xpred09=[]
xpred1=[]
xpre0=[]
xpre01=[]
xpre03=[]
xpre05=[]
xpre07=[]
xpre09=[]
xpre1=[]
x00=[]
x003=[]
x005=[]
x009=[]
x01=[]
x030=[]
x0303=[]
x0305=[]
x0309=[]
x031=[]
x050=[]
x0503=[]
x0505=[]
x0509=[]
x051=[]
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


for index in P2.keys():
    a=P2[index].split("(")
    b=a[1].split(")")
    PrdS=b[0]
    c=a[2].split(")")
    PreS = c[0]
    if PrdS == "0":

        xpred0.append(x[index])
    if PrdS == "0.1":
        xpred01.append(x[index])
    if PrdS == "0.3":
        xpred03.append(x[index])

    if PrdS == "0.5":
        xpred05.append(x[index])
    if PrdS == "0.7":
        xpred07.append(x[index])

    if PrdS == "0.9":
        xpred09.append(x[index])

    if PrdS == "1":
        xpred1.append(x[index])



    if PreS == "0":

        xpre0.append(y[index])
    if PreS == "0.1":
        xpre01.append(y[index])

    if PreS == "0.3":
        xpre03.append(y[index])

    if PreS == "0.5":
        xpre05.append(y[index])
    if PreS == "0.7":
        xpre07.append(y[index])

    if PreS == "0.9":
        xpre09.append(y[index])

    if PreS == "1":
        xpre1.append(y[index])







xpred0=sum(xpred0)
xpred01=sum(xpred01)
xpred03=sum(xpred03)
xpred05=sum(xpred05)
xpred07=sum(xpred07)
xpred09=sum(xpred09)
xpred1=sum(xpred1)
xpre0=sum(xpre0)
xpre01=sum(xpre01)
xpre03=sum(xpre03)
xpre05=sum(xpre05)
xpre07=sum(xpre07)
xpre09=sum(xpre09)
xpre1=sum(xpre1)

x=[0,0.1,0.3,0.5,0.7,0.9,1]
pred=[xpred0/140,xpred01/140,xpred03/140,xpred05/140,xpred07/140,xpred09/140,xpred1/140]
prey=[xpre0/140,xpre01/140,xpre03/140,xpre05/140,xpre07/140,xpre09/140,xpre1/140]
print(xpred0,xpred03,xpred05,xpred07,xpred09,xpred1)
print(xpre0,xpre03,xpre05,xpre07,xpre09,xpre1)

# plt.subplot(2, 1,1)
# plt.scatter(x,pred,c="#A9561E",label='Predator')
plt.plot(x,pred,"#A9561E",marker='*')
plt.legend(["Predator"],loc='upper left')
plt.xlabel('Sociality regime')
plt.ylabel('Average rewards')
# plt.subplot(2, 1,2)
plt.show()
plt.plot(x,prey,"#6495ED",marker='*')
plt.legend(["Prey"],loc='upper left')
plt.xlabel('Sociality regime')
plt.ylabel('Average rewards')
# plt.plot(x, pred,color="#A9561E",x, prey,color="#6495ED",marker=‘*’)
# plt.legend(loc='lower  right')
# plt.subplot(2, 1,2)
# plt.scatter(x,prey,c="#6495ED",label='Prey')
# plt.plot(x, prey,color="#6495ED",linewidth=2,label='Prey')
# plt.legend(loc='lower  right')
plt.show()