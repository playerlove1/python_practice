#使用sklearn  的Decision Tree
#Example from:  http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree


#建立訓練資料點
#X =座標  Y=類別
X = [[0, 0], [1, 1]]
Y = [0, 1]
#初始化分類器
clf = tree.DecisionTreeClassifier()
#fit  等同於訓練的意思 (給定  training data)
clf = clf.fit(X, Y)
#預測結果
print(clf.predict([[2., 2.]]))

print(clf.predict_proba([[2., 2.]]))