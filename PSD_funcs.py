import numpy as np


def is_PSD(X):
    '''
    Function to test positive semidefinitiness of symmetric matrices.
    '''
    # If absolute value of eigen-value is below 10^-15 it is set zero
    #  More efficient implementation can follow https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    E = np.linalg.eigvalsh(X) 
    E[np.abs(E) < 1e-15] = 0
    return np.all(E >= 0)

def PSD2normedCorr(K):
    '''
    Function to convert PSD matrices to unit trace normalized Correlation matrices
    '''

    assert(len(K.shape) == 2), "input array must be square"
    assert(K.shape[0] == K.shape[1]), "Input matrix must be square"
    N = K.shape[0]
    Diag = np.diag(K)
    Diag_nonzeros = Diag > 0
    K_Corr = np.zeros(K.shape)
    sqrt_inv_nz_Diag = np.zeros(Diag.shape)
    sqrt_inv_nz_Diag[Diag_nonzeros] = 1 / np.sqrt(Diag[Diag_nonzeros])
    K_Corr[Diag_nonzeros, :] = K[Diag_nonzeros, :] * sqrt_inv_nz_Diag[Diag_nonzeros][:, np.newaxis]
    K_Corr[:, Diag_nonzeros] = K_Corr[:, Diag_nonzeros] * sqrt_inv_nz_Diag[Diag_nonzeros][np.newaxis, :]
    np.fill_diagonal(K_Corr, 1)
    K_Corr = (K_Corr + K_Corr.T)/2

    return K_Corr/N


def Dist2Corr(D):
    '''
    This function convert distance matrices to Correlation operators.
    '''
    assert(len(D.shape) == 2), "input array must be square"
    assert(D.shape[0] == D.shape[1]), "Input matrix must be square"
    # Centering
    HD = D - np.mean(D, axis=0)
    HDH = HD - np.mean(HD, axis=1)[:, np.newaxis]
    # Eigenvalue Truncation
    [E, U] = np.linalg.eigh(-HDH)
    E[E < 0] = 0
    K = U@np.diag(E)@U.T
    # Normalization for PSD to Correlation matrix
    Diag = np.diag(K)
    Diag_nonzeros = Diag > 0
    K_Corr = np.zeros(K.shape)
    sqrt_inv_nz_Diag = np.zeros(Diag.shape)
    sqrt_inv_nz_Diag[Diag_nonzeros] = 1 / np.sqrt(Diag[Diag_nonzeros])
    K_Corr[Diag_nonzeros, :] = K[Diag_nonzeros, :] * sqrt_inv_nz_Diag[Diag_nonzeros][:, np.newaxis]
    K_Corr[:, Diag_nonzeros] = K_Corr[:, Diag_nonzeros] * sqrt_inv_nz_Diag[Diag_nonzeros][np.newaxis, :]
    np.fill_diagonal(K_Corr, 1)
    K_Corr = (K_Corr + K_Corr.T)/2

    return K_Corr
