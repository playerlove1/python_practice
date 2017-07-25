import numpy as np



class Perceptron(object):
    """Perceptron 感知器分類器　　只有在(1)兩類是線性可分的情況下　且　(2)學習速率很小情況下才會收斂
    
    參數命名規則:不會被初始化的屬性 會在屬性名字後加上_
    
    
    參數
    --------
    eta: float 
        Learning rate學習速率　(0.0~1.0)
    n_iter: int 
        迭代次數(設定epochs避免永不停止的情況)
    
    屬性
    --------
    w_: 1維陣列
    
    errors_: list
      紀錄各迭代回合分類錯誤的數目
    
    """
    #建構式
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter=n_iter
    def fit(self, X, y):
        """Fit training data
        參數
        ---
        X: {array-like}, shape = {n_samples, n_features}
            訓練特徵(feature)
        y: {array-like} shape = {n_samples}
            預測目標(target)
        
        
        
        Returns
        -------
        self : object
        """
        #初始權重數目  y=w0+w1X1+w2X2 ...    +1代表加入bias (即w0)
        #w_為一個 1列3行的矩陣
        self.w_= np.zeros(1+X.shape[1])
        #顯示初始權重
        # print('perceptron: 初始權重w_:'+np.array_str(self.w_))
        self.errors_ = []
        
        #針對每個迭代回合
        for _ in range(self.n_iter):
            errors = 0
            #利用zip將X與y結合成一個tuple的list   ex.  a=[1,2,3] b=[4,5,6]   zip(a,b)=[(1,4),(2,5).(3,6)]
            #xi=[x1,x2]   target=y   zip(X,y)=[x1,x2,target]
            #針對每個樣本進行迴圈
            for xi, target in zip(X, y):
               # 印出每個樣本
               # print("xi:"+str(xi)+" target:"+str(target))
               
               #更新的權重=學習速率*(該筆資料的target-該筆資料的預測結果)
               #若分類錯誤就修正　　若沒分類錯誤就不動(0)
               update = self.eta * (target - self.predict(xi))
               
               #w1以後的部分更新
               self.w_[1:] += update * xi
               #w0的更新  (bias)
               self.w_[0:] += update
               #若update !=0  則會+1　否則+0
               errors += int(update != 0.0)
               
               # print("update:"+str(update))
            #紀錄當次迭代的分類錯誤數
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        """
        計算net input z=w0X0+w1X1+w2X2=w^T X =(向量內積)
        """
        return np.dot(X, self.w_[1:] )+ self.w_[0]
    def predict(self, X):
        """回傳類別標籤"""
        #如果計算出來的net_input(淨輸入)　>=0的時候為1
        return np.where(self.net_input(X) >= 0.0, 1, -1)