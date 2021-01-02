import numpy as np
import random
import matplotlib.pyplot as plt




class SMO_SVM_Classifier:
    def __init__(self, F, x, y, hyperparameter, max_passes, C, kernel):
        """
        ::input:: kernel - either "RBF" or "Polynomial"
        """
        self.x = x
        self.y = y
        self.hyperparameter = hyperparameter
        self.max_passes = max_passes
        self.C = C
        self.F = F
        self.kernel = kernel
        
        self._set_kernel()
        
        
    def train(self):
        self.alpha_star, self.b_star = self._Simplified_SMO(self.x, self.y, self.hyperparameter, self.max_passes, self.C)
        
        
    def accuracy(self, test_x, test_y):
        """
        input: N x d data matrix
        """
        acc_store = []
        
        for i in range(len(test_x)):
            acc_store.append(self._prediction(self.x, test_x[i,:], self.hyperparameter, self.alpha_star, self.y , self.b_star))
        return np.sum(acc_store == test_y)/len(test_y)
        
    def plot_objective_value(self):
        plt.plot(self.hist)


    
        
    def _Simplified_SMO(self, X, y, d, max_passes, C):
        n = len(X[:,0])
        alpha = np.zeros((n))
        b = 0

        passes = 0

        self.hist = []
        while (passes < max_passes):
            num_changed_alphas = 0
            prev_alpha = alpha.copy()
            

            for i in range(n):

                x_i,  y_i= X[i,:],  y[i]
                E_i = self._prediction(X, x_i, d, alpha, y, b) - y_i
                if ((y_i * E_i < -0.005) and (alpha[i] < C)) or ((y_i * E_i > 0.005) and (alpha[i] > 0)):
                    
                    print('Iter {}: f* = {}'.format(i, self.F._f0(alpha[np.newaxis].T)))
                    self.hist.append(self.F._f0(alpha[np.newaxis].T).item())

                    
                    


                                        
                    
                    j = random.randint(0,n-1)
                    while i == j:
                        j=random.randint(1,n-1)
                    x_j, y_j =  X[j,:], y[j]
                    E_j = self._prediction(X, x_j, d, alpha, y, b) - y_j

                    old_alpha_i, old_alpha_j = alpha[i].copy(), alpha[j].copy()

                    if y_i == y_j:
                        L , H = max(0, old_alpha_i + old_alpha_j - C), min(C, old_alpha_i + old_alpha_j)
                    else:
                        L , H = max(0, old_alpha_i - old_alpha_j), min(C, C + old_alpha_i - old_alpha_j)
                    if L == H:
                        continue

                    eta =  2 * self._kernel2(x_i,x_j,d) - self._kernel2(x_i, x_i,d) - self._kernel2(x_j, x_j, d)
                    if eta >= 0:
                        continue        


                    
                    alpha[j] -= float(y_j * (E_i - E_j))/eta
                    
                    if alpha[j] > H:
                        alpha[j] == H
                    elif L <= alpha[j] <= H:
                        alpha[j] = alpha[j]
                    elif alpha[j] < L:
                        alpha[j] = L


                    alpha[i] += y_i*y_j * (old_alpha_j - alpha[j])

                    b1 = b - E_i - y_i * (alpha[i] - old_alpha_i) * self._kernel2(x_i, x_i,d) - y_j * (alpha[j] - old_alpha_j) * self._kernel2(x_i, x_j, d)
                    b2 = b - E_j - y_i * (alpha[i] - old_alpha_i) * self._kernel2(x_i, x_j,d) - y_j * (alpha[j] - old_alpha_j) * self._kernel2(x_j, x_j,d)

                    if 0 < alpha[i] and C > alpha[i]:
                        b = b1
                    elif 0 < alpha[j] and C > alpha[j]:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas += 1
    

            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        print(np.dot(alpha, y))
        return alpha, b
        
        
        
    def _set_kernel(self):
        if self.kernel == "RBF":
            self._kernel = self._RBF_kernel
            self._prediction = self._predict_for_RBF
            self.gamma = self.hyperparameter
            self._kernel2 = self._RBF_kernel
            
        elif self.kernel == "Polynomial":
            self._kernel = self._polynomial_kernel
            self._prediction = self._prediction_for_polynomial
            self.d = self.hyperparameter
            self._kernel2 = self._poly_kernel2
            
        else:
            raise NameError ("Kernel is not defined")
            
            
            
    def _poly_kernel2(self,x1, x2, d):
        return np.dot(x1, x2)**d
            
            
    def _polynomial_kernel(self, x, x_prime, d):
        x_prime = (x_prime).reshape(len(x_prime),1)
        return np.dot(x, x_prime)**d
    
    def _RBF_kernel(self, x1, x2, gamma):
#         axis = None if x2.ndim==1 else 1
        return np.exp(-gamma*np.linalg.norm(x1-x2)**2)
    
    

    
    
    def _predict_for_RBF(self, x, x_prime, gamma, alpha, y, b):
        size = x.shape[0]
        y_pred = np.zeros(size)
            
        alpha = alpha.reshape(len(alpha),1)
        y = y.reshape(len(y), 1)
        
        for i in range(x.shape[0]):
            y_pred[i] = alpha[i]*y[i]*self._RBF_kernel(x[i], x_prime, gamma) + b
        return np.sign(np.sum(y_pred))
        
        
        
    def _prediction_for_polynomial(self,x, x_prime, d, alpha, y, b):
        cal_kernel = self._kernel(x, x_prime, d)
        cal_kernel = cal_kernel.reshape(len(cal_kernel),1)
        
        alpha = alpha.reshape(len(alpha),1)
        y = y.reshape(len(y), 1)

        prediction = np.sign(np.matmul((alpha * y).T, cal_kernel) + b)
        return prediction[0][0]