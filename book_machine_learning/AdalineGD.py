import numpy as np
class AdalineGD (object):
    """ADAptive Linear Neuron Classifier
        
        Parameters (參數)
        ------------
        eta: float
             Learning rate學習速率　(0.0~1.0)
        n_iter: int
            迭代次數
        
        Attribute(屬性)
        ------------
        w_ : 1d-array
            Weights after fitting  (fitting後的權重)
        errors_: list
            每輪分類錯誤的個數
        cost_ :list
                每回合所計算出的成本函數
        """
        
    #建構式
    def __init__ (self, eta=0.01, n_iter=50):
            self.eta=eta
            self.n_iter=n_iter
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
        self.w_= np.zeros(1+X.shape[1])
        #
        self.cost_ = []
        #針對每個迭代回合   
        for i in range(self.n_iter):
            #計算net_input list(將所有樣本的預測值都計算出來)
            output = self.net_input(X)
            #計算errors list(將所有樣本的誤差都算出來) (實際-預測)
            errors = (y-output)
            #考慮整個訓練集的資料更新權重
            #X.T.dot(errors) = np.dot(X.T,errors)   = (純python法) sum([i*j for i,j in zip(X,errors)])
            self.w_[1:]+=self.eta*X.T.dot(errors)  
            self.w_[0]+=self.eta*errors.sum()
            #cost function 成本函數  誤差的平方*1/2
            cost= (errors**2).sum() /2.0
            #將該回合的成本函數 加到所有回合統計的list
            self.cost_.append(cost)
        return self
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
        

      