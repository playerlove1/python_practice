""" Edited-Nearest Neighbors Regression (ENNR) Instance Selection 的方法"""
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsRegressor

class ENNR():
    #建構式 (default k=3)
    def __init__(self, alpha, model, n_neighbors=3):
        self.model = model
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.classifier = None
    def reduce_data(self, X, y):
        if self.classifier == None:
            self.classifier = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            self.classifier.fit(X,y)
        if self.classifier.n_neighbors != self.n_neighbors:
            self.classifier.n_neighbors = self.n_neighbors
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #如果Knn的K 大於等於 資料的筆數  則不減少資料  將全部回傳
        if self.n_neighbors >= len(X):
            self.X_ = np.array(X)
            self.y_ = np.array(y)
            self.reduction_ = 0.0
            return self.X_, self.y_
            
        #初始樣本子集的遮罩(預設全被都挑出來)
        mask = np.ones(y.size, dtype=bool)
        #依序移除每個樣本的遮罩
        tmp_m = np.ones(y.size, dtype=bool)
        
        #依序拿掉每個樣本
        for i in range(y.size):
            tmp_m[i] = not tmp_m[i]
            #用拿掉第i個樣本的資料進行迴歸模型的訓練
            self.model.fit(X[tmp_m], y[tmp_m])
            #第i個樣本的資料
            feature, target = X[i], y[i]
            #計算迴歸模型對第i個樣本的預測資料
            y_pred = self.model.predict(feature.reshape(1, -1))
            #門檻值
            threshold = self.alpha * self.get_neighbors_std(feature.reshape(1, -1), y = y) 
            
            #如果拿掉第i個樣本樣本的迴歸大於門檻值將會被視為分類錯誤被移除 
            if abs(target-y_pred) > threshold:
                mask[i] = not mask[i]
            
            #再將第i個樣本補回去
            tmp_m[i] = not tmp_m[i]

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        #print("reduction_X:"+str(self.X_ ))
        #print("eduction_y:"+str(self.y_))
        #print("reduction:"+str(self.reduction_))
        return self.X_, self.y_
        
    #利用KNN取得最近鄰居的標準差
    def get_neighbors_std(self, X, y):
        mask = np.zeros(y.size, dtype=bool)
        neighbor = self.classifier.kneighbors(X)[1][0]
        for i in neighbor:
            mask[i] = not mask[i]
        
        return np.std(y[mask])