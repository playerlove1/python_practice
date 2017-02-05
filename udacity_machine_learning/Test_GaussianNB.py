#使用sklearn  的Naive Bayes
#Example from:  http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
import numpy as np 

#建立訓練資料點
#X =座標  Y=類別
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

#import 高斯素樸貝氏
from sklearn.naive_bayes import GaussianNB
#初始化分類器
clf = GaussianNB()
#fit  等同於訓練的意思 (給定  training data)
clf.fit(X, Y)
 
#預測 -0.8,-1 此點的類別
print(clf.predict([[-0.8, -1]]))
 

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
 print(clf_pf.predict([[-0.8, -1]]))