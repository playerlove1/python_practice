import numpy as np
from enn_regression import ENNR
from DROP2RE import DROP2RE
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class DROP3RE():
    """"Decremental Reduction Optimization Procedure3RE(DROP3) 
    ENNR()->初始排序->DROP2RE()
    
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
    def __init__(self,model, alpha=0.1, n_neighbors=3):
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
        # print('----DROP3RE----')        
        #先利用ENNR過濾資料
        ennr = ENNR(alpha = self.alpha, model = self.model , n_neighbors = self.n_neighbors)
        X_enn, y_enn=ennr.reduce_data(X,y)
        #初始排序
        X_enn, y_enn=self.init_sorting(X_enn, y_enn)
        
        #將過濾後的資料利用DROP2RE
        drop2re=DROP2RE(model = self.model, n_neighbors = self.n_neighbors)
        self.X_, self.y_=drop2re.reduce_data(X_enn,y_enn)
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP3RE reduction:"+str(self.reduction_))
        # print('----DROP3RE----')
        return self.X_, self.y_
        
    #利用KNN取得最近鄰居的標準差
    def get_neighbors_std(self, X, y):
        mask = np.zeros(y.size, dtype=bool)
        neighbor = self.classifier.kneighbors(X)[1][0]
        for i in neighbor:
            mask[i] = not mask[i]
        
        return np.std(y[mask])

    #依照最近敵人距離降冪排序
    def init_sorting(self, X,y):
        '''
        迴歸型的最近敵人的距離被定義為: 兩者的誤差大於threshold就視為是敵人
        '''
        #將KNN分類器fit樣本
        self.classifier.fit(X, y)
        #全部算與最近鄰居距離的list
        distance_list=[]
        
        #初始排序(最近敵人的距離被定義為:  兩者的誤差大於threshold就視為是敵人)
        for i in range(y.size):
            #計算門檻值
            threshold = self.alpha * self.get_neighbors_std(X[i].reshape(1, -1), y = y) 
            
            min = float("inf")
            #每個樣本搜尋符合最近敵人的樣本
            for j in range(y.size):
                #滿足被視為敵人的條件
                if(abs(y[i]-y[j])>threshold):
                    #找到距離最近的
                    if(min>(abs(y[i]-y[j])-threshold)):
                        min =abs(y[i]-y[j])-threshold
            #將每個樣本的最近敵人之距離加到list中
            distance_list.append(min)
        
        #將原始資料按照最近敵人距離的遠近作降冪排序  (是一個 index list)
        sorted_list=sorted(range(len(distance_list)), key=lambda k:distance_list[k], reverse=True)
        
        #將原始資料按照最近敵人距離的遠近作降冪排序
        X=X[sorted_list]
        y=y[sorted_list]
        return X, y