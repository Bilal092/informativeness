from numpy.linalg import norm as norm
from numpy import trace as trace
from numpy import mean as mean
import numpy as np
import scipy as sp

#__all__ = ['PSD_Distances']
'''
This module works specfically for trace normalized correlation operators 

'''

def sqrtm_safe(K):
    [E, U] = sp.linalg.eigh((K + K.T)/2)
    E[E < 0] = 0
    Lambd = np.diag(np.sqrt(E))
    return U @ Lambd @ U.T


def PSD_distances(A,B, dist_type):
    assert(A.shape == B.shape), "matrices must of same size"

    if dist_type == "bures" or dist_type == "Bures":

        A = A/np.trace(A)
        B = B/np.trace(B)
        sqrt_A = sqrtm_safe(A)
        C = sqrtm_safe(sqrt_A@B@sqrt_A)
        #temp = np.trace(A + B) - 2 * np.trace(C) for more general PSD Matrices
        temp = 2 - 2 * np.trace(C)
        return temp
    
    if dist_type == "Cosine" or dist_type == "cosine":
        normA = norm(A)
        normB = norm(B)

        assert (normA > 0), "First input matrix is Zero norm"
        assert (normB > 0), "Second input matrix is Zero norm "
        
        Frob_inner_product = np.sum(A*B)

        return 2 - 2*Frob_inner_product/(normA*normB)
