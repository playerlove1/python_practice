"""Condensed-Nearest Neighbors (CNN)  Instance Selection 的方法"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsRegressor

import random

class CNNR():
    """"Condensed-Nearest Neighbors Regression(CNNR) 
        
        每個類別都需要一些代表性的樣本  並且測試其他樣本可以正確的分類   移除那些多餘的INSTANCE  保留在決策邊界的instance
        (小於threshold)
    參數
    --------
    init_size: int
        初始化挑的樣本數
    model : classifier
        用來計算迴歸的模型
    n_neighbors: int
        使用KNN的K值
    alpha: float
        判定threshold的參數alpha  (用幾個標準差)


    
    屬性
    --------
    model : classifier
        用來計算迴歸的模型
    classifier: classifier
        所使用的分類器(即KNN)
    
    classes_ : list
        目前的資料集當中  Target的unique (ex. [a,b,c]) 
    X_:  list
        feature
    y_:  list
        target
        
    """
    
    
    #建構式 (default k=3)
    def __init__(self, alpha, model,init_size = 2, n_neighbors=3):
        self.init_size = init_size
        self.model = model
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.classifier = None

    #
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier =KNeighborsRegressor(n_neighbors=self.n_neighbors)
            self.classifier.fit(X,y)
            
        subset_X = []
        subset_y = []
        
        #所有資料的出現與否的mask 預設每個資料都尚未被挑選
        mask = np.zeros(y.size, dtype=bool)
        
        #針對所有樣本隨機挑init_size個數的樣本
        rand = random.sample(range(mask.shape[0]),self.init_size)
        #亂數挑到的樣本加入subset_X
        for i in rand:
            subset_X = subset_X + [X[i]]
            mask[i] = not mask[i]
            #print("subset_X:"+str(subset_X))
            subset_y  = subset_y  + [y[i]]
            #print("subset_y:"+str(subset_y))

        #拿剛剛隨機的樣本進行初始回歸模型的fit
        self.model.fit(subset_X, subset_y )
        
        #針對所有樣本
        for feature, target in zip(X, y):
            #計算預測值
            y_pred =  self.model.predict(feature.reshape(1,-1))
            #門檻值
            threshold = self.alpha * self.get_neighbors_std(feature.reshape(1, -1), y = y) 
            
            #大於門檻值 (相當於分類的分類錯誤)
            if abs(target-y_pred) > threshold :
                subset_X = subset_X + [feature]
                subset_y = subset_y + [target]
                #重新fit 回歸模型
                self.model.fit(subset_X, subset_y)

        self.X_ = np.asarray(subset_X)
        self.y_ = np.asarray(subset_y)
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
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

        

