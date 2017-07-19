from cnn import CNN
from enn import ENN
from renn import RENN
from rnn import RNN
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],[4,1],[-1,1] ])
# y = np.array([1, 1, 1, 2, 2, 2, 2, 2])
# cnn = CNN()
# cnn.reduce_data(X, y)
# print(cnn.predict([[-0.8, -1]]))


#讀取資料
    #使用sklearn的dataset
    #透過iris.target取出y  並且取出前100筆資料
    #target_name=['setosa' 'versicolor' 'virginica']  因此對應y的0,1,2    將setosa 轉為-1  其餘的轉為1   (由於y只取0:100  因此只有setosa與versicolor兩種)
    #feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   取出其中的sepal length (cm)跟'petal length (cm)' 作為feature
    #取出資料集當中的前100筆資料當中的0,2的部分的行向量作為X
iris = datasets.load_iris()
y=iris.target
X=iris.data[0:150,[0,2]]
# y=iris.target[0:100]
# y = np.where(y == 0, -1, 1)
# X=iris.data[0:100,[0,2]]
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

plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

#顯示類別圖
plt.title('orginal')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

''' 

#CNN

cnn = CNN()
reduce_X, reduce_y = cnn.reduce_data(X, y)


# type1=np.where(reduce_y == -1)
# print(str(type1))
# type2=np.where(reduce_y == 1)
# print(str(type2))
type1=np.where(reduce_y == 0)
print(str(type1))
type2=np.where(reduce_y == 1)
print(str(type2))
type3=np.where(reduce_y == 2)

# 前50個會是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 後50個是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# 後50個是另一類
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='versicolor')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('reduce')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
'''


#ENN
enn = ENN()

reduce_X, reduce_y =enn.reduce_data(X,y)


type1=np.where(reduce_y == 0)

type2=np.where(reduce_y == 1)

type3=np.where(reduce_y == 2)

# 1~50是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 50~100是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# 100~150是另一類
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='versicolor')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

print('len:'+str(len(reduce_X)))
#顯示類別圖
plt.title('reduce enn')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


'''

#RENN
renn = RENN()

reduce_X, reduce_y =renn.reduce_data(X,y)


type1=np.where(reduce_y == 0)

type2=np.where(reduce_y == 1)

type3=np.where(reduce_y == 2)

# 1~50是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 50~100是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# 100~150是另一類
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='versicolor')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')


#顯示類別圖
plt.title('reduce renn')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
'''
'''
rnn = RNN()

reduce_X, reduce_y =rnn.reduce_data(X,y)


type1=np.where(reduce_y == 0)

type2=np.where(reduce_y == 1)

type3=np.where(reduce_y == 2)

# 1~50是同類
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 50~100是另一類
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# 100~150是另一類
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='versicolor')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')


#顯示類別圖
plt.title('reduce rnn')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

'''