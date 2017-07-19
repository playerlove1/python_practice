""" Repeated Edited-Nearest Neighbors (RENN) Instance Selection 的方法"""
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

from enn import ENN

class RENN():
    #建構式 (default k=3)
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None
        
    def reduce_data(self, X, y):

        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #該dataset的類別list  (ex. [a,b,c])
        classes = np.unique(y)
        self.classes_ = classes      
        
        enn=ENN(n_neighbors=self.n_neighbors)
        
        
        hold_X_,hold_y_,r_ =  X, y, 1.0
        count = 0 
        #直到上次的enn的結果與前一次相同時停止(即不再減少樣本)
        while r_ !=0 :            
            enn.reduce_data(hold_X_,hold_y_)
            hold_X_= enn.X_
            hold_y_= enn.y_
            r_=enn.reduction_
            count = count+1
            
        self.X_ = hold_X_
        self.y_ = hold_y_
        self.reduction_ = 1.0 - float(hold_y_.shape[0]) / y.shape[0]
        print("迭代次數:"+str(count))
        
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
