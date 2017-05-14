#測試
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from perceptron import Perceptron
from decision_bondary import plot_decision_regions
# from plot_decision_regions import plot_decision_regions
from AdalineGD  import AdalineGD
from AdalineSGD  import AdalineSGD
#讀取資料
    #使用sklearn的dataset
    #透過iris.target取出y  並且取出前100筆資料
    #target_name=['setosa' 'versicolor' 'virginica']  因此對應y的0,1,2    將setosa 轉為-1  其餘的轉為1   (由於y只取0:100  因此只有setosa與versicolor兩種)
    #feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   取出其中的sepal length (cm)跟'petal length (cm)' 作為feature
    #取出資料集當中的前100筆資料當中的0,2的部分的行向量作為X
iris = datasets.load_iris()
y=iris.target[0:100]
y = np.where(y == 0, -1, 1)
X=iris.data[0:100,[0,2]]
    #使用pandas 載入UCI ML Repository
    #回傳的資料為DataFrame 其中包含 Fature與Target  (Feature:0~3 Target:4)
    #取出資料集當中的前100筆資料當中的4的部分的行向量作為y
    #將y的字串為Iris-setosa的 轉為  -1 其餘的轉為 1 (由於y只取0:100  因此只有setosa與versicolor兩種)
    #取出資料集當中的前100筆資料當中的0,2的部分的行向量作為X
# iris=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# y = iris.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = iris.iloc[0:100,[0,2]].values

# 前50個會是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 後50個是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')



#顯示類別圖
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#Perceptron
ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Number of misclassifications")
plt.show()

#決策邊界
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()


#(使用subplots)
#畫adalineGD不同學習速率的圖   
fig, ax =plt.subplots(1,2, figsize=(8,4))

#第一種learning rate的情況
ada1=AdalineGD(n_iter=10, eta=0.01).fit(X,y)
#x軸為不同回合的結果  y軸為取log後的cost 
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_title('Adaline - Learning rate 0.01')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-square-error)')

#第二種learning rate的情況      
#x軸為不同回合的結果  y軸為取log後的cost 
ada2=AdalineGD(n_iter=10, eta=0.0001).fit(X,y)   
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_title("Adaline - Learning rate 0.0001") 
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum-square-error')
plt.show() 





#(使用subplots)
#畫決策邊界跟成本下降的曲線圖 (AdalineGD)
fig, ax =plt.subplots(1,2, figsize=(12,6))

#決策邊界
#使用特徵縮放 (標準化:讓特徵值滿足標準常態分配 Standard normal distribution 並且每個特徵的平均值都是0)
    #使用NumPy的mean與std完成  (xj'=(xj-mean(X))/std(X))
X_std=np.copy(X)
# x1標準化
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
# x標準化
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
   
ada=AdalineGD(n_iter=15, eta=0.01).fit(X_std,y)


plot_decision_regions(X_std, y, classifier=ada,ax=ax[0])
ax[0].set_title("Adaline-Gradient Descent") 
ax[0].set_ylabel('petal length [cm][standardlized]')
ax[0].set_xlabel('sepal length [cm][standardlized]')
ax[0].legend(loc='upper left')

#成本下降曲線圖 (AdalineGD)
ax[1].plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
ax[1].set_title("Adaline - Learning rate 0.01 [standardlized]") 
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum-square-error')


plt.show()



#(使用subplots)
#畫決策邊界跟成本下降的曲線圖 (AdalineSGD)
fig, ax =plt.subplots(1,2, figsize=(12,6))

#決策邊界
ada=AdalineSGD(n_iter=15, eta=0.01).fit(X_std,y)
plot_decision_regions(X_std, y, classifier=ada,ax=ax[0])
ax[0].set_title("Adaline-Stochastic Gradient Descent") 
ax[0].set_xlabel('sepal length [cm][standardlized]')
ax[0].set_ylabel('petal length [cm][standardlized]')
ax[0].legend(loc='upper left')

#成本下降曲線圖 (AdalineGD)
ax[1].plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
ax[1].set_title("Adaline - Learning rate 0.01 [standardlized] SGD") 
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Average Cost')

plt.show()
