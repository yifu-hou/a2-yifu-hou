import numpy as np
import scipy.stats as sts
from numba.pycc import CC


S = 1000
T = 4160
rho, mu, sigma = 0.5, 3.0, 1.0
z_0 = mu

np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

cc = CC('test_aot')

@cc.export('simulate_aot', 'f8[:,:](f8[:,:], f4, f4)')
def simulate_aot(z_mat, rho, mu):
    '''
    Compute S simulations of a lifetime T - AOT ACCELERATED
    Inputs:
        z_mat: (numpy) container for simulation results
        rho: (float) degree of persistence
        mu: long-run average
    Return:
        z_mat: (numpy) simulation result of size (T, S)
    '''

    T, S = z_mat.shape
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

    return z_mat

cc.compile()