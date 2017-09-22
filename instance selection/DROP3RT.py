import numpy as np
from enn_regression import ENNR
from DROP2RT import DROP2RT
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class DROP3RT():
    """"Decremental Reduction Optimization Procedure3RT(DROP3) 
    ENNR()->DROP2RT()
    
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    alpha :  int or float
        ENNR取得所使用之alpha
    
    model : classifier 
         迴歸模型
    屬性
    --------
    X_:  list
        feature
    y_:  list
        target
        
    """
    #建構式 (default k=3)
    def __init__(self, model, alpha=0.1, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.alpha = alpha
        self.model = model
    
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)
        # print('----DROP3RT----')        
        #先利用ENNR過濾資料
        ennr = ENNR(alpha = self.alpha, model = self.model , n_neighbors = self.n_neighbors)
        X_enn, y_enn=ennr.reduce_data(X,y)
     
        #將過濾後的資料利用DROP2RE
        drop2rt=DROP2RT(alpha=self.alpha ,model = self.model, n_neighbors = self.n_neighbors)
        self.X_, self.y_=drop2rt.reduce_data(X_enn,y_enn)
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP3RT reduction:"+str(self.reduction_))
        # print('----DROP3RT----')
        return self.X_, self.y_
        
    #利用KNN取得最近鄰居的標準差
    def get_neighbors_std(self, X, y):
        mask = np.zeros(y.size, dtype=bool)
        neighbor = self.classifier.kneighbors(X)[1][0]
        for i in neighbor:
            mask[i] = not mask[i]
        
        return np.std(y[mask])
