import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class DROP1():
    """"Decremental Reduction Optimization Procedure1(DROP1) 
    
    with:子集S 訓練knn 對子集S分類的正確個數
    without:子集S/樣本x 訓練knn 對子集S分類的正確個數 (/:集合的差集)
    if without>= with:
        移除樣本x
    
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    
    屬性
    --------
    classifier: classifier
        所使用的分類器(即KNN)
    X_:  np array
        feature 
    y_:  np array
        target
    reduction_: float
        reduced sample size/all sample size
        
    """
    #建構式 (default k=1)
    def __init__(self,n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.classifier = None
    
    
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)

        #依序拿掉每個樣本的mask
        mask = np.ones(y.size, dtype=bool)
        #紀錄那些在是被拿掉的
        drop_mask=np.ones(y.size, dtype=bool)

        for i in range(y.size):

            #拿子集樣本  進行Knn的fit
            self.classifier.fit(X[drop_mask], y[drop_mask])
            #預測結果
            pred=self.classifier.predict(X[drop_mask])
            #預測正確的個數
            correct_with = np.sum([y[drop_mask]==pred])
            # print("with:",correct_with)
            
            #依序拿掉每個樣本
            mask=drop_mask.copy()
            mask[i] = not mask[i]
            self.classifier.fit(X[mask], y[mask])
            #預測結果
            pred=self.classifier.predict(X[drop_mask])
            #預測正確的個數
            correct_without =np.sum([y[drop_mask]==pred])
            # print("without:",correct_without)
            
            #如果拿掉第i個樣本樣本的KNN 可以正確分類第i個樣本 則在mask中將該index所在的值改為True  
            if (correct_without-correct_with)>=0:
                drop_mask[i] = not drop_mask[i]
            
            #再將第i個樣本補回去
            mask[i] = not mask[i]
        self.X_ = np.asarray(X[drop_mask])
        self.y_ = np.asarray(y[drop_mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP1 reduction:"+str(self.reduction_))
        return self.X_, self.y_


