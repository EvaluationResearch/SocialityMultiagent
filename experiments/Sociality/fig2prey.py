# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import statsmodels.formula.api as smf

from scipy.stats import pearsonr
def main():
    # tips = sns.load_dataset("C:/Users/hp/maddpg/experiments/Sociality/Eprey.csv")
    dfA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/Aprey.csv')

    dfS = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/Sprey.csv')

    dfE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/Eprey.csv')

    # xy = range(20)
    #
    # a=dfE['x']

    plt.xlabel('Prey rewards(in thousands)',size=20)
    plt.ylabel('Predator rewards(in thousands)',size=20)
    plt.legend(loc='upper right')

    # x = ["8", "11", "14", "17", "20", "23", "26", "29", "32", "35"]
    # x = ["15", "30", "45", "60", "75", "90", "105", "120", "135",""]

    cm = plt.cm.get_cmap('RdYlBu')
    [A1, A2] = pearsonr(dfA['x'], dfA['y'])
    [S1, S2] = pearsonr(dfS['x'], dfS['y'])
    [E1, E2] = pearsonr(dfE['x'], dfE['y'])

    # sns.scatterplot(dfE['x'], dfE['y'],c=dfE['x'])
    plt.scatter(dfE['x'], dfE['y'], marker='v', label=u"Rewards from egalitarian regime", s=20)
    plt.scatter(dfS['x'], dfS['y'], marker='*', color='#dd85d7', label='Rewards from selfish regime',s=70)
    plt.scatter(dfA['x'], dfA['y'], marker='o', color='#a9be70', label='Rewards from altruistic regime',s=30)
    plt.legend(loc='upper right')
    # plt.colorbar(sc)

    resultE = smf.ols('y~x', data=dfE).fit()
    resultS = smf.ols('y~x', data=dfS).fit()
    resultA= smf.ols('y~x', data=dfA).fit()
    print(resultE.params)
    print(resultE.summary())
    print(resultS.params)
    print(resultS.summary())
    print(resultA.params)
    print(resultA.summary())
    y_fittedE = resultE.fittedvalues
    y_fittedS = resultS.fittedvalues
    y_fittedA = resultA.fittedvalues
    #
    # a = plt.subplots(figsize=(10, 10))
    plt.plot(dfE['x'], y_fittedE, 'b', label='OLS for egalitarian rewards')
    plt.plot(dfS['x'], y_fittedS, 'm', label='OLS for selfish rewards')
    plt.plot(dfA['x'], y_fittedA, 'olive', label='OLS for altruistic rewards')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    main()

