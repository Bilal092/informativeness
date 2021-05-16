import numpy as np
import scipy as sp

def sqrtm_safe(K):
    [E,U] = sp.linalg.eigh( (K + K.T) / 2 )
    E[E < 0] = 0
    Lambd = np.diag( np.sqrt( E) )
    return U @ Lambd @ U.T

def Na(a,n):
    return a + (1 - a) * n / (n-1) * (np.eye(n) - 1/n)

def Info(K, metric):
    assert(len(K.shape) == 2), "Input must be a matrix"
    assert(K.shape[0] == K.shape[1]), "Matrix must be square"

    n = K.shape[0]
    k_bar =  np.mean(K)
  
    if metric == "Bures"  or metric == "bures" :
        if n > 1:
            HK = K - np.mean(K, axis=0)
            HKH = HK - np.mean(HK, axis=1)[:, np.newaxis]
            HKH = (HKH + HKH.T) /2
            sqrt_HKH = sqrtm_safe( HKH )
            factorK = (np.trace(sqrt_HKH))**2 / (n**2 - n)
            temp = (1 - np.sqrt(k_bar + factorK))
            return temp
        else:
            return 0

    if metric == "Cosine" or metric == "cosine":
        if n > 1:
            # copied matlab code from paper's code
            K = K / np.trace(K) * n
            K_norm = np.sqrt(np.mean(K*K))
            temp = 1- (1 / K_norm)*np.sqrt((n*k_bar**2 - 2*k_bar+1) / (n-1))
            return temp
        else:
            return 0


# This part is written for computation of Infromativess when K = Z*Z.T, where Z belongs to R^{n*d} with 
def Info_embeddings(Z, metric):
    # In contrast to informtivness paper our matrix Z is R^{n*d}
    n = Z.shape[0]
    K = Z@Z.T

    if metric == "Bures" or metric == "bures":
        if n > 1:
            Z1 = np.sum(Z, axis=0)
            Z1_norm_sqrd = (np.linalg.norm(Z1))**2
            HZ_norm_sqrd = (np.linalg.norm(Z - np.mean(Z, axis=0), ord = 'nuc'))**2 
            temp = 1 - np.sqrt(Z1_norm_sqrd / n**2 + 1/(n**2 - n) * HZ_norm_sqrd)
            return temp
        else:
            return 0
    
    if metric == "Cosine" or metric == "cosine":
        if n > 1:
            Z1 = np.sum(Z, axis=0)
            Z1_norm_sqrd = (np.linalg.norm(Z1))**2
            Z_fro_sqrd = (np.linalg.norm(Z, ord = 'fro'))**2
            a = (1/n) * (Z1_norm_sqrd / Z_fro_sqrd)
            N = Na(a, n)
            temp = 1 - np.sum(K*N)/(np.linalg.norm(N) * np.linalg.norm(K))
            return temp
        else:
            return 0

