import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier

class DROP2():
    """"Decremental Reduction Optimization Procedure2(DROP2) 
    初始排序:依照最近敵人距離降冪排序
    with:子集S 訓練knn 對全部資料分類的正確個數
    without:子集S/樣本x 訓練knn 對全部資料分類的正確個數 (/:集合的差集)
    if without>= with:
        移除樣本x
    
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    
    屬性
    --------
    classifier: classifier
    
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
        self.classifier_without = None
        
    
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)
        
        #初始排序
        X, y=self.init_sorting(X, y)
        #遮罩使其依序拿掉樣本
        mask = np.ones(y.size, dtype=bool)
        #紀錄那些在是被拿掉的
        drop_mask=np.ones(y.size, dtype=bool)
        
        for i in range(y.size):
            #拿子集樣本  重新進行Knn的fit
            self.classifier.fit(X[drop_mask], y[drop_mask])
            #knn分類的預測結果(對全部樣本)
            pred=self.classifier.predict(X)
            #with分類正確的個數
            correct_with = np.sum([y==pred])
            # print("with:",correct_with)
            
            #子集拿掉目前的樣本X
            mask=drop_mask.copy()
            mask[i] = not mask[i]
            #以拿掉目前樣本X的資料集 進行knn的fit
            self.classifier.fit(X[mask], y[mask])
            #預測結果
            pred=self.classifier.predict(X)
            #without分類正確的個數
            correct_without =np.sum([y==pred])
            # print("without:",correct_without)
            
            #如果 without >= with (分類正確的個數) 則拿掉樣本X(將drop_mask中該index 的值設為false) 
            if (correct_without-correct_with)>=0:
                drop_mask[i] = not drop_mask[i]
            
            #再將第i個樣本補回去
            mask[i] = not mask[i]
        self.X_ = np.asarray(X[drop_mask])
        self.y_ = np.asarray(y[drop_mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP2 reduction:"+str(self.reduction_))
        return self.X_, self.y_
    
    #依照最近敵人距離降冪排序
    def init_sorting(self, X,y):
        #用來拿掉與目前樣本同類別的遮罩
        mask = np.ones(y.size, dtype=bool)
        #全部與最近敵人距離的list
        distance_list=[]
        
        #找出依照最近敵人距離降冪排序的資料集
        for i in range(y.size):
            #遮罩  拿掉與目前該筆資料一樣類別的資料
            mask = y !=y[i]
            #將目前該筆資料回填
            mask[i] = True
            #用剩餘資料來訓練(即只有一個是 i這筆資料的類別  其他都是enemy)
            self.classifier.fit(X[mask], y[mask])
            #找最近鄰居(除了自己)  即為最近敵人
            n=self.classifier.kneighbors(X[i].reshape(1, -1))[0][0][1]
            #將最近敵人的距離加到list
            distance_list.append(n)

        #將原始資料按照最近敵人距離的遠近作降冪排序  (是一個 index list)
        sorted_list=sorted(range(len(distance_list)), key=lambda k:distance_list[k], reverse=True)
        
        #將原始資料按照最近敵人距離的遠近作降冪排序
        X=X[sorted_list]
        y=y[sorted_list]
        return X,y

