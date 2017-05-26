"""Condensed-Nearest Neighbors (CNN)  Instance Selection 的方法"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier



class CNN():
    """"Condensed-Nearest Neighbors (CNN) 
        
        每個類別都需要一些代表性的樣本  並且測試其他樣本可以正確的分類   移除那些多餘的INSTANCE  保留在決策邊界的instance
    
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    
    屬性
    --------
    classifier: classifier
        所使用的分類器(即KNN)
    
    classes_ : list
        目前的資料集當中  Target的unique (ex. [a,b,c]) 
    X_:  list
        feature
    y_:  list
        target
        
    """
    
    
    #建構式 (default k=1)
    def __init__(self,n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.classifier = None
    #
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        subset_X = []
        subset_y = []
        
        #該dataset的類別list  (ex. [a,b,c]) 
        classes = np.unique(y)
        self.classes_ = classes
        
        
        #針對每個在類別list裡的類別進行迴圈
        for cur_class in classes:
            #遮罩  將y與目前在迴圈的類別進行比較  若一樣則為True 若不一樣則為False
            mask = y == cur_class
            print("mask:"+str(mask))
            #取出為True的那些資料  (即與目前迴圈相同類別的instance)
            insts = X[mask]
            print("X[mask]:"+str(X[mask]))
            
            #由0~被挑出的那些資料中 亂數取一個 加入subset_X
            subset_X = subset_X + [insts[np.random.randint(0, insts.shape[0])]]
            print("subset_X:"+str(subset_X))
            #subset_y 加入目前執行的instance之類別
            subset_y  = subset_y  + [cur_class]

        #拿剛剛隨機的樣本  進行Knn的fit
        self.classifier.fit(subset_X, subset_y )
        
        #針對所有樣本
        for feature, target in zip(X, y):
            #如果跟分類器裡的不一致
            if self.classifier.predict(feature) != [target]:
                #就把該點加入prots_s
                subset_X = subset_X + [feature]
                #labels_s加入目前執行的instance之類別
                subset_y = subset_y + [target]
                #重新fit knn
                self.classifier.fit(subset_X, subset_y)
        
        
        self.X_ = np.asarray(subset_X)
        self.y_ = np.asarray(subset_y)
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        print("reduction_X:"+str(self.X_ ))
        print("eduction_y:"+str(self.y_))
        print("reduction:"+str(self.reduction_))
        
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
