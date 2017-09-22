import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors.classification import KNeighborsClassifier


class DROP2RE():
    """"Decremental Reduction Optimization Procedure2 RE(DROP2) 
    DROP2 for regression 沒有使用初始排序(根據前人研究結果  使用的效果會比較差)
    
    ewith :計算x樣本的鄰居(nn) 進行迴歸模型的訓練後的誤差總和
           針對x樣本的每個鄰居a 以迴歸模型(nn/a 訓練)進行預測 (/:集合的差集)
           計算誤差並加到ewith中
    ewithout: 計算x樣本的鄰居(nn 排除x) 進行迴歸模型的訓練後的誤差總和
           針對樣本的每個鄰居a 以迴歸模型(以 nn/x,a 訓練)進行預測 (/:集合的差集)
           計算誤差並加到ewithout中
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
    def __init__(self, model, n_neighbors=3):
        self.model = model
        self.n_neighbors = n_neighbors
        self.classifier = None

    def reduce_data(self, X,y):
        #確定 X與y是 否是符合規定  X 2d  y 1d y不能是np.nan 也不能是np.inf  
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        #建立KNN 分類器
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)

        #紀錄那些在是被拿掉的遮罩  預設全為true 若符合drop的條件 則會被改為False
        drop_mask=np.ones(y.size, dtype=bool)
        

        #針對每個樣本x進行迴圈
        for i in range(y.size):
            #將KNN分類器fit 子集樣本
            self.classifier.fit(X[drop_mask], y[drop_mask])
            #找出目前樣本x的最近鄰居nn list
            nn=self.classifier.kneighbors(X[i].reshape(1, -1))[1][0]
            #將x樣本排除於nnlist  (nn = nn/x)
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
                #將先前拿到掉的樣本x補回nn_mask中
                nn_mask[i]=not nn_mask[i]
                # print(np.sum(nn_mask),np.sum(nn_mask_without))
                
                #計算ewith
                
                #以nn_mask (nn/a    /:集合的差集)訓練迴歸模型
                self.model.fit(X[nn_mask], y[nn_mask])
                #取得迴歸模型預測
                pred=self.model.predict(X[nn_mask])
                #計算誤差總和
                ewith += np.sum(y[nn_mask]-pred)
                
                #計算ewithout
                #以nn_mask_without (nn/a,x    /:集合的差集)訓練迴歸模型
                self.model.fit(X[nn_mask_without], y[nn_mask_without])
                #取得迴歸模型預測
                pred=self.model.predict(X[nn_mask])
                #計算誤差總和
                ewithout +=np.sum(y[nn_mask]-pred)
                
            # print("with:", ewith)
            # print("without:", ewithout)
            #如果 ewithout<=ewith 的話就拿掉樣本x (將drop_mask中該index 的值設為false)
            if ewithout<=ewith :
                drop_mask[i] = not drop_mask[i]
        
        self.X_ = np.asarray(X[drop_mask])
        self.y_ = np.asarray(y[drop_mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        
        # print("reduction_X:"+str(self.X_ ))
        # print("eduction_y:"+str(self.y_))
        # print("DROP2RE reduction:"+str(self.reduction_))
        return self.X_, self.y_


