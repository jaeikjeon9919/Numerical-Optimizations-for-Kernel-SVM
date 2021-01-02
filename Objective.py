import numpy as np

class f:
    def __init__(self, alpha_0, D, C, t=1e4):
        self.alpha = alpha_0
        self.D = D
        self.t = t
        self.C = C
        self.one = np.ones_like(alpha_0)
        self.assign_value()
        
    def _f0(self, alpha):
        f0 = 0.5  * alpha.T @ self.D @ alpha -  np.sum(alpha) 
        return f0
        

    def _f(self, alpha):
        main_obj = 0.5 * self.t * alpha.T @ self.D @ alpha - self.t * np.sum(alpha)
        loss = - np.sum(np.log(alpha)) - np.sum(np.log(self.C * self.one - alpha))
        return main_obj + loss
    
    def _df(self, alpha):
        main_obj_grad = self.t * self.D @ alpha - self.t * self.one
        loss_grad = - 1/alpha - 1/(alpha-self.C * self.one)
        return main_obj_grad + loss_grad
    
    def _d2f(self, alpha):
        main_obj_hess = self.t * self.D
        loss_hess = np.diag((1/alpha**2).flatten()) + np.diag((1/(alpha-self.C)**2).flatten())
        return main_obj_hess + loss_hess
    
    
    def assign_value(self):
        self.f = self._f(self.alpha)
        self.df = self._df(self.alpha)
        self.d2f = self._d2f(self.alpha)
    
    def update_alpha(self, alpha):
        self.alpha = alpha
        self.assign_value()
    
    def update_t(self, t):
        self.t = t
        self.assign_value()
    
#     def update_f0