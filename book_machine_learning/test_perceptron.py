#測試
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from perceptron import Perceptron
from decision_bondary import plot_decision_regions

iris = datasets.load_iris()
# iris=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# print(iris.header)
#取出前100個 鳶尾花的類別標籤
# y = iris.iloc[0:100, 4].values
y=iris.target

#將標籤字串轉為  -1(setosa) or 1(versicolor)
# y = np.where(y == 'Iris-setosa', -1, 1)
y = np.where(y == 0, -1, 1)

#取出資料集當中的0,2作為feature
# X = iris.iloc[0:100,[0,2]].values
X=iris.data[0:100,[0,2]]

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Number of misclassifications")
plt.show()

plot_decision_regions(X, y, classifier=ppn)

