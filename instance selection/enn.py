""" Edited-Nearest Neighbors (ENN) Instance Selection 的方法"""
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class ENN():
    #建構式 (default k=3)
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None
    def reduce_data(self, X, y):
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        if self.classifier.n_neighbors != self.n_neighbors:
            self.classifier.n_neighbors = self.n_neighbors
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #該dataset的類別list  (ex. [a,b,c])
        classes = np.unique(y)
        self.classes_ = classes      
        #如果Knn的K 大於等於 資料的筆數  則不減少資料  將全部回傳
        if self.n_neighbors >= len(X):
            self.X_ = np.array(X)
            self.y_ = np.array(y)
            self.reduction_ = 0.0
            return self.X_, self.y_
        #遮罩  
        mask = np.zeros(y.size, dtype=bool)
        #暫存目前的
        tmp_m = np.ones(y.size, dtype=bool)
        
        #依序拿掉每個樣本
        for i in range(y.size):
            tmp_m[i] = not tmp_m[i]
            #用拿掉第i個樣本的dataset做 knn
            self.classifier.fit(X[tmp_m], y[tmp_m])
            #第i個樣本的資料
            feature, target = X[i], y[i]
            
            #如果拿掉第i個樣本樣本的KNN 可以正確分類第i個樣本 則在mask中將該index所在的值改為True  
            if self.classifier.predict(feature) == [target]:
                mask[i] = not mask[i]
            
            #再將第i個樣本補回去
            tmp_m[i] = not tmp_m[i]

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("reduction:"+str(self.reduction_))
        return self.X_, self.y_
    #利用目前的分類器進行預測
    def predict(self, X):

        X = check_array(X)
        if not hasattr(self, "X_") or self.X_ is None:
            raise AttributeError("Model has not been trained yet.")

        if not hasattr(self, "y_") or self.y_ is None:
            raise AttributeError("Model has not been trained yet.")

        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        self.classifier.fit(self.X_, self.y_)
        return self.classifier.predict(X)
