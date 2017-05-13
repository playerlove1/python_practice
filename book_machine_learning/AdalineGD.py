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
        self.w_= np.zeros(1+X.shape[1])
        self.cost_ = []
            
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            self.w_[1:]=self.eta*X.T.dot(errors)
            self.w_[0]=self.eta*errors.sum()
            #cost function 成本函數  誤差的平方*1/2
            cost= (errors**2).sum() /2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        """Caculate net input(計算網路的淨輸入)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, X):
        """Compute linear activation 線性的activate function"""
        return self.net_input(X)
        
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X)>= 0.0 ,1 -1)
        

      