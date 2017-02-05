#使用sklearn  的SVM
#Example from:http://scikit-learn.org/stable/modules/svm.html
from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
#初始化分類器
clf = svm.SVC()
#給定訓練資料
clf.fit(X, y) 
#輸出預測結果
print(clf.predict([[2., 2.]]))
#輸出support vectors
# get support vectors
print(clf.support_vectors_)
# get indices of support vectors
print(clf.support_)
# get number of support vectors for each class
print(clf.n_support_)