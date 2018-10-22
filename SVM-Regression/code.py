import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[2]].values

# *** feature scaling is important to use SVM
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X,y)

y_pred = sc_y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X,y,color = 'red')
