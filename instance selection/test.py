from cnn import CNN
from enn import ENN
from renn import RENN
from DROP1 import DROP1
from DROP2 import DROP2
from DROP3 import DROP3
from cnn_regression import CNNR
from enn_regression import ENNR
from DROP2RE import DROP2RE
from DROP2RT import DROP2RT
from DROP3RE import DROP3RE
from DROP3RT import DROP3RT
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsRegressor
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],[4,1],[-1,1] ])
# y = np.array([1, 1, 1, 2, 2, 2, 2, 2])
# cnn = CNN()
# cnn.reduce_data(X, y)
# print(cnn.predict([[-0.8, -1]]))


'''
#讀取資料
    #使用sklearn的dataset
    #透過iris.target取出y  並且取出前100筆資料
    #target_name=['setosa' 'versicolor' 'virginica']  因此對應y的0,1,2    將setosa 轉為-1  其餘的轉為1   (由於y只取0:100  因此只有setosa與versicolor兩種)
    #feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   取出其中的sepal length (cm)跟'petal length (cm)' 作為feature
    #取出資料集當中的前100筆資料當中的0,2的部分的行向量作為X
iris = datasets.load_iris()
y=iris.target
X=iris.data[0:150,[0,2]]
y=iris.target[0:100]
y = np.where(y == 0, -1, 1)
X=iris.data[0:100,[0,2]]
    #使用pandas 載入UCI ML Repository
    #回傳的資料為DataFrame 其中包含 Fature與Target  (Feature:0~3 Target:4)
    #取出資料集當中的前100筆資料當中的4的部分的行向量作為y
    #將y的字串為Iris-setosa的 轉為  -1 其餘的轉為 1 (由於y只取0:100  因此只有setosa與versicolor兩種)
    #取出資料集當中的前100筆資料當中的0,2的部分的行向量作為X
iris=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = iris.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = iris.iloc[0:100,[0,2]].values
'''
#完整資料集
iris = datasets.load_iris()
y=iris.target
X=iris.data[0:150,[0,2]]

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
subset_X = []
subset_y = []
        
#所有資料的出現與否的mask 預設每個資料都尚未被挑選
mask = np.zeros(y.size, dtype=bool)

#針對所有樣本隨機挑init_size個數的樣本
rand = random.sample(range(mask.shape[0]),2)
#亂數挑到的樣本加入subset_X
for i in rand:
    subset_X = subset_X + [X[i]]
    mask[i] = not mask[i]
    print("subset_X:"+str(subset_X))
    subset_y  = subset_y  + [y[i]]
    print("subset_y:"+str(subset_y))
index=0
for row in X:
    #print("row:",row,"sub:",subset_X[0])
    
    if np.array_equal( row,np.asarray(subset_X[0])):
        print("index:",index)
    index += 1
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(X,y)
neighbor=knn.kneighbors(np.asarray(subset_X[0]))[1][0]

print(knn.kneighbors(np.asarray(subset_X[0]))[1][0])
y_mask=np.zeros(y.size, dtype=bool)
for i in neighbor:
    y_mask[i] = not y_mask[i]

print("sum:",y[y_mask])
print("std:",np.std([1,2,3,4,5]))
'''
# 分類example

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
#DROP1
drop1 = DROP1()
reduce_X, reduce_y = drop1.reduce_data(X, y)
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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP1 ')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#DROP2
drop2 = DROP2()
reduce_X, reduce_y = drop2.reduce_data(X, y)
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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP2 ')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#DROP3
drop3 = DROP3()
reduce_X, reduce_y = drop3.reduce_data(X, y)
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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP3 ')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()




#迴歸example
'''
#CNNR
knn=KNeighborsRegressor(n_neighbors=2)
cnnr = CNNR(alpha=1, model=knn)
reduce_X, reduce_y = cnnr.reduce_data(X, y)


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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('CNNR ')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
'''

'''
#ENNR
knn=KNeighborsRegressor(n_neighbors=2)
ennr = ENNR(alpha=0.1, model=knn)
reduce_X, reduce_y = ennr.reduce_data(X, y)

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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('ENNR')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
'''

#DROP2RE
knn=KNeighborsRegressor(n_neighbors=2)
drop2re=DROP2RE(knn)
reduce_X, reduce_y = drop2re.reduce_data(X, y)

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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP2RE')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#DROP3RE
knn=KNeighborsRegressor(n_neighbors=2)
drop3re=DROP3RE(knn)
reduce_X, reduce_y = drop3re.reduce_data(X, y)

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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP3RE')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


#DROP2RT
knn=KNeighborsRegressor(n_neighbors=2)
drop2rt=DROP2RT(knn)
drop2rt.reduce_data(X,y)
reduce_X, reduce_y = drop2rt.reduce_data(X, y)

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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP2RT')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


#DROP3RT
knn=KNeighborsRegressor(n_neighbors=2)
drop3rt=DROP3RT(knn)
reduce_X, reduce_y = drop3rt.reduce_data(X, y)

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
plt.scatter(X[100:150, 0], X[100:150, 1], color='black', marker='s', label='virginica')

plt.scatter(reduce_X[type1, 0], reduce_X[type1, 1], color='green', marker='*', label='is setosa')
plt.scatter(reduce_X[type2, 0],reduce_X[type2, 1], color='yellow', marker='*', label='is versicolor')
plt.scatter(reduce_X[type3, 0],reduce_X[type3, 1], color='pink', marker='*', label='is virginica')

#顯示類別圖
plt.title('DROP3RT')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()