import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier


class DROP2RT():
    """"Decremental Reduction Optimization Procedure2 RT(DROPRT) 
    
    與DROP2RE類似  但並不是直接計算其誤差值 而是設定Threshold來判斷
    若大於門檻值則+1 (有視為一個分類錯誤的之意)
    
    ewith : 一個樣本x所有+1的結果
    ewithout: 一個樣本x所有+1的結果
        if ewithout<= ewith:
            移除樣本x
    參數
    --------
    n_neighbors: int
        使用KNN的K值
    model : classifier 
         迴歸模型
    
    
    屬性
    --------
    classifier: classifier
        所使用的分類器(即KNN) 
    X_:  list
        feature
    y_:  list
        target
        
    """
    #建構式 (default k=3)
    def __init__(self, model, alpha=0.1, n_neighbors=3):
        self.model = model
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.alpha = alpha
    
    
    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)
        #初始排序
        X, y = self.init_sorting(X, y)

        #紀錄那些在是被拿掉的遮罩  預設全為true 若符合drop的條件 則會被改為False
        drop_mask=np.ones(y.size, dtype=bool)

        #針對每個樣本x進行迴圈
        for i in range(y.size):
            #將KNN分類器fit 子集樣本
            self.classifier.fit(X[drop_mask], y[drop_mask])
            #計算門檻值
            threshold = self.alpha * self.get_neighbors_std(X[i].reshape(1, -1), y = y) 
            #找出目前樣本x的最近鄰居nn list
            nn=self.classifier.kneighbors(X[i].reshape(1, -1))[1][0]
            #將x樣本排除於nnlist
            nn= nn[nn != i ]
            #初始化ewithout 與ewith
            ewithout =0
            ewith =0

            #根據每個NN list中的元素做運算
            for j in nn:
                #without mask  初始化 (全為false)
                nn_mask_without=np.zeros(y.size, dtype=bool)
                #with mask 初始化 (全為false)
                nn_mask=np.zeros(y.size, dtype=bool)

                #依照nn list 拿掉目前要計算的鄰居a (當不是鄰居a的時候 會被設定為true)
                for k in range(nn.size):
                    #如果不是目前的鄰居a 則挑出
                    if(nn[k]!=j):
                        nn_mask_without[nn[k]] = not nn_mask_without[nn[k]]
                        nn_mask[nn[k]] = not nn_mask[nn[k]]
                #將先前拿到掉的樣本x補回  nn_mask中
                nn_mask[i]=not nn_mask[i]
                # print(np.sum(nn_mask),np.sum(nn_mask_without))
                #計算ewith
                
                #以nn_mask (nn/a    /:集合的差集)訓練迴歸模型
                self.model.fit(X[nn_mask], y[nn_mask])
                #取得迴歸模型預測
                pred=self.model.predict(X[nn_mask])
                #計算誤差
                if abs(np.sum(y[nn_mask]-pred)) <= threshold:
                    ewith+=1
                
                #計算ewithout
                #以nn_mask_without (nn/a,x    /:集合的差集)訓練迴歸模型
                self.model.fit(X[nn_mask_without], y[nn_mask_without])
                #取得迴歸模型預測
                pred=self.model.predict(X[nn_mask])
                #計算誤差
                if abs(np.sum(y[nn_mask]-pred)) <= threshold:
                    ewithout+=1
                
                
            # print("with:", ewith)
            # print("without:", ewithout)
            #如果 ewithout>=ewith 的話就拿掉樣本x (將drop_mask中該index 的值設為false)
            if ewithout>=ewith :
                drop_mask[i] = not drop_mask[i]
        
        self.X_ = np.asarray(X[drop_mask])
        self.y_ = np.asarray(y[drop_mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP2RT reduction:"+str(self.reduction_))
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