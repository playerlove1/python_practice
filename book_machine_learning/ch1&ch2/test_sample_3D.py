import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt


#讀取資料
    #使用sklearn的dataset
    #透過iris.target取出y  並且取出前100筆資料
    #target_name=['setosa' 'versicolor' 'virginica']  因此對應y的0,1,2    將setosa 轉為-1  其餘的轉為1   (由於y只取0:100  因此只有setosa與versicolor兩種)
    #feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   取出其中的petal length (cm)跟 petal width (cm) 作為feature
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

#使用sklearn的cross_validation模組的train_test_split函數  將原始訓練資料切割為訓練集與測試集  (將資料切成30%測試 70%訓練)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#使用sklearn的preprocessing模組的StandardScaler來做標準化 (標準化:讓特徵值滿足標準常態分配 Standard normal distribution 並且每個特徵的平均值都是0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#sklearn中的分類演算法  大多利用 一對餘(One-vs.-Rest,OvR)的方法來支援多元分類
ppn = Perceptron(n_iter=40, eta0=0.01, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('錯誤分類的樣本數(Misclassified samples):%d' % (y_test != y_pred).sum())
print('精準度(Accuracy): %.2f' % accuracy_score(y_test,y_pred))

# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.vstack((y_train, y_test))
plot_decision_regions(X=X_train_std, y = y_train, clf=ppn, X_highlight = X_test_std)
plt.xlabel('petal length [cm] [Standardized]')
plt.ylabel('pepal width [cm] [Standardized]')
plt.legend(loc='upper left')
plt.show()