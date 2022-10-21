# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import statsmodels.formula.api as smf

from scipy.stats import pearsonr
def main():
    dfA = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/A.csv')

    dfS = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/S.csv')

    dfE = pd.read_csv('C:/Users/hp/maddpg/experiments/Sociality/E.csv')
    # table = xlrd.open_workbook('1.csv')
    # pdE = table.sheet_by_name("pdE")
    # pdS = table.sheet_by_name("pdS")
    # pdA = table.sheet_by_name("pdA")
    # row_count = pdE.nrows
    # column_count = pdE.ncols
    #
    # roE = np.zeros(row_count-1)
    # coE = np.zeros(row_count-1)
    # roS = np.zeros(row_count - 1)
    # coS = np.zeros(row_count - 1)
    # roA = np.zeros(row_count - 1)
    # coA = np.zeros(row_count - 1)
    # k=0
    # for i in range(1, row_count):
    #
    #     roE[k] = int(pdE.cell(i,0).value)
    #     coE[k] = int(pdE.cell(i,1).value)
    #     roS[k] = int(pdS.cell(i, 0).value)
    #     coS[k] = int(pdS.cell(i, 1).value)
    #     roA[k] = int(pdA.cell(i,0).value)
    #     coA[k] = int(pdA.cell(i, 1).value)
    #     k+=1

    [A1,A2]=pearsonr(dfA['x'],dfA['y'])
    [S1, S2] = pearsonr(dfS['x'], dfS['y'])
    [E1, E2] = pearsonr(dfE['x'], dfE['y'])
    plt.xlabel('Predator rewards(in thousands)',size=20)
    plt.ylabel('Prey rewards(in thousands)',size=20)
    plt.legend(loc='upper right')

    # x = ["8", "11", "14", "17", "20", "23", "26", "29", "32", "35"]
    # x = ["15", "30", "45", "60", "75", "90", "105", "120", "135",""]


    plt.scatter(dfE['x'], dfE['y'], marker='v', color='#0066CC', label=u"Rewards from egalitarian regime", s=20)

    plt.scatter(dfS['x'], dfS['y'], marker='*', color='#dd85d7', label='Rewards from selfish regime', s=70)
    plt.scatter(dfA['x'], dfA['y'], marker='o', color='#a9be70', label='Rewards from altruistic regime', s=30)
    plt.legend(loc='upper right')
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xticklabels(["7","5","4","2","1"], rotation='vertical', fontsize=18)
    # plt.scatter(3170, -1917, marker='d', color='k',  s=30)
    #
    # plt.scatter(5712, -3103, marker='d', color='k',  s=50)
    # plt.scatter(1360, -1092, marker='d', color='k',  s=15)
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

