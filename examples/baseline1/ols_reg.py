import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.loadtxt("yhoo_hist.csv", delimiter=',', skiprows=1, usecols=range(1,7))
Y = np.array([a1 / a2 - 100 for a1, a2 in zip(X[1:], X)])
X = X[:-1,:] # .reshape(-1,1)
X_train = X[:-100]
X_test = X[-100:]
Y_train = Y[:-100]
Y_test = Y[-100:]
clf = linear_model.LinearRegression()
clf.fit (X_train, Y_train)
print 'Coefficients: ', clf.coef_
print 'Residual sum of squares: %.2f' % np.mean((clf.predict(X_test) - Y_test) ** 2)
print 'Variance score: %.2f' % clf.score(X_test, Y_test)
#print clf.predict(X_test)
#print Y_test
#plt.scatter(X_test, Y_test,  color='black')
#plt.plot(X_test, clf.predict(X_test), color='blue', linewidth=3)
#plt.xticks(())
#plt.yticks(())
#plt.show()
