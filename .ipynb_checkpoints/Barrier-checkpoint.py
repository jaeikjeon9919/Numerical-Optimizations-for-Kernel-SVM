import numpy as np

def Barrier(F, Centering_step, t_0, m, mu, epsilon):
    iteration = 0 
    t = t_0
    hist = []
    hist2 = []
    while m/t > epsilon:
        
        alpha_star = Centering_step.descent_line_search()
        t = mu * t
        
        F.update_alpha(alpha_star)
        F.update_t(t)
        Centering_step.update_F(F)
        
        iteration += 1
        print('Iter {}: f* = {}'.format(iteration+1, F.f))
        
        hist.append(F.f.item())
#         print(mu)
    return F.alpha, hist
