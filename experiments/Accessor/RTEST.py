import pandas as pd
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data=pd.read_excel(r"accessor.xls")

X=data.drop(labels='R',axis=1)
y=data.R

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.08,random_state=0)
#X_train
#y_train

model=DecisionTreeRegressor(random_state=0)
model.fit(X_train,y_train)

y=model.score(X_test,y_test)

pred=model.predict(X_test)
print('\n',pd.DataFrame({'Prediction':pred,'Real':y_test}))