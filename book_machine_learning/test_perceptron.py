#測試
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from perceptron import Perceptron
from decision_bondary import plot_decision_regions

#讀取資料
iris = datasets.load_iris()
# iris=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# print(iris.header)
#取出前100個 鳶尾花的類別標籤
# y = iris.iloc[0:100, 4].values



# 取前100個
y=iris.target[0:100]



#將y=0 的轉為-1  其餘的轉為1
y = np.where(y == 0, -1, 1)
#將標籤字串轉為  -1(setosa) or 1(versicolor)
# y = np.where(y == 'Iris-setosa', -1, 1)




#取出資料集當中的0,2作為feature
X=iris.data[0:100,[0,2]]
# X = iris.iloc[0:100,[0,2]].values

#前50個會是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#後50個是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

#顯示類別圖
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#Perceptron每輪的錯誤分類數
ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Number of misclassifications")
plt.show()


#決策邊界
plot_decision_regions(X, y, classifier=ppn)

