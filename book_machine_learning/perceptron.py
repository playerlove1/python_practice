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
      分類錯誤的
    
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
        self.w_= np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                
                self.w_[1:] += update * xi
                self.w_[0:] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        """
        計算net input
        """
        return np.dot(X, self.w_[1:] )+ self.w_[0]
    def predict(self, X):
        """回傳類別標籤"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)