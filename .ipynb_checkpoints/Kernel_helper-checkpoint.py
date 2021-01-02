import numpy as np

def polynomial_kernel(x1, x2, d):
#     print(x1.shape)
    return np.dot(x1,x2) ** d


def rbf_kernel(x1, x2, gamma):
    
    axis = None if x2.ndim==1 else 1
    return np.exp(-gamma*np.linalg.norm(x1-x2, axis=axis) ** 2)




def calc_D(x, y, hyperparameter, kernel_type):
    if kernel_type == "Polynomial":
        kernel = polynomial_kernel
    elif kernel_type == "RBF":
        kernel = rbf_kernel
    x_dim = x.shape[0]
    D = np.zeros((x_dim, x_dim))
    for i in range(x_dim):
        for j in range(x_dim):
            D[i,j] = y[i] * y[j] * kernel(x[i], x[j], hyperparameter)
    return D






