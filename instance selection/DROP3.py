#Decremental Reduction Optimization Procedure(DROP)
import numpy as np
from enn import ENN
from DROP2 import DROP2
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class DROP3():
    """"Decremental Reduction Optimization Procedure3(DROP3) 
    先用ENN再用DROP2()
    
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    
    屬性
    --------
    X_:  list
        feature
    y_:  list
        target
        
    """
    #建構式 (default k=1)
    def __init__(self,n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None
    
    
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        # print('----DROP3----')
        #呼叫ENN過濾資料
        enn = ENN(self.n_neighbors)
        X_enn, y_enn=enn.reduce_data(X,y)
        #將過濾後的資料交給Drop2
        drop2=DROP2(self.n_neighbors)
        self.X_, self.y_=drop2.reduce_data(X_enn,y_enn)
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP3 reduction:"+str(self.reduction_))
        # print('----DROP3----')
        return self.X_, self.y_


