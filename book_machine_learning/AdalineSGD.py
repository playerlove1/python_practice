import numpy as np
from numpy.random import seed

class AdalineSGD (object):
    """ADAptive Linear Neuron Classifier
        
        Parameters (參數)
        ------------
        eta: float
             Learning rate學習速率　(0.0~1.0)
        n_iter: int
            迭代次數
        
        Attribute(屬性)
        ------------
        _w : 1d-array
            Weights after fitting  (fitting後的權重)
        w_initialized :bool
            使否經過初始化
        errors_: list
            每輪分類錯誤的個數
        cost_ :list
            每回合所計算出的成本函數
        shuffle: bool (default:True)
            每輪都打亂樣本  如果是True就會避免重複
        random_state int (default:None)
            設定亂數 以供suffling打亂用  初始化權重
        """
        
    #建構式
    def __init__ (self, eta=0.01, n_iter=50,shuffle=True, random_state=None):
            self.eta=eta
            self.n_iter=n_iter
            self.w_initialized = False
            self.shuffle = shuffle
            
            if random_state:
                seed(random_state)
    def fit (self, X, y):
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
        self._initialize_weights(X.shape[1])
        
        self.cost_ = []
        
        #針對每個迭代回合   
        for i in range(self.n_iter):
            #如果是就打亂
            if self.shuffle:
                X, y = self._shuffle(X, y)
                
            cost = []
        
            #針對每個樣本進行迴圈
            for xi, target in zip(X, y):
                #計算cost的值
                cost.append(self._update_weights(xi, target))
            #計算所有樣本平均的Cost值
            avg_cost = sum(cost) / len(y)
            #將該次迭代平均的Cost值加入cost_
            self.cost_.append(avg_cost)
        return self
    #online 
    #對個別樣本呼叫partial_fit  ex. ada.partial_fit(X_std[0,:],y[0])
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self        
    #隨機重排    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    #初始化權重
    def _initialize_weights(self, m):
        self.w_= np.zeros(1 + m)
        self.w_initialized = True
    #更新權重
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        # cost = 0.5 * error**2
        cost =(error**2) /2.0
        return cost
        
    def net_input(self, X):
        """Caculate net input(計算網路的淨輸入)  (即向量內積)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, X):
        """Compute linear activation 也就是 =net_input的作用函數"""
        return self.net_input(X)
        
    def predict(self, X):
        """Return class label after unit step"""
        #如果計算出來的net_input(淨輸入)　>=0的時候為1
        return np.where(self.activation(X)>= 0.0 ,1 ,-1)
        

      