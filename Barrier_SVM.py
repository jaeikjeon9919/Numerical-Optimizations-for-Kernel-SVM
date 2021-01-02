import numpy as np
from Kernel_helper import rbf_kernel

 


class Barrier_SVM_Classifier:
    def __init__(self, alpha, D, x, y, gamma):
        self.alpha = alpha
        self.D = D
        self.x = x
        self.y = y
        self.gamma = gamma
        
        self.b = self.Calc_bias()
        
        
        
    def Calc_bias(self):
        sup_vec = np.where(self.alpha > 1e-5)[0] # find support vectors
        b = 0
        for j in sup_vec:
            b += self.y[j] - np.sum(self.alpha*self.y*self.D[:,j])
        return b / sup_vec.size
    
    def predict(self, x):
        size = x.shape[0]
        y_pred = np.zeros(size)
        if x.ndim==1:
            return np.sign(np.sum(self.alpha*self.y*rbf_kernel(x, self.x, self.gamma)) + self.b)
        else:
            for i in range(size):
                y_pred[i] = np.sign(np.sum(self.alpha*self.y*rbf_kernel(x[i], self.x, self.gamma)) + self.b)
            return y_pred
        
    def accuracy(self, x, y_true):
        y_pred = self.predict(x)
        return np.sum(y_pred==y_true) / y_true.size