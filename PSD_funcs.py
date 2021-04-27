import numpy as np


def is_PSD(X):
    return np.all(np.linalg.eigvals(X) >= 0)


def PSD2normedCorr(K):

    assert(len(K.shape) == 2), "input array must be square"
    assert(K.shape[0] == K.shape[1]), "Input matrix must be square"

    N = K.shape[0]
    Diag = np.diag(K)
    Diag_nonzeros = Diag > 0
    D = np.sqrt(Diag)[:, np.newaxis]
    K_Corr = np.zeros(K.shape)
    K_Corr[Diag_nonzeros, :] = K[Diag_nonzeros, :]/D[Diag_nonzeros]
    K_Corr[:, Diag_nonzeros] = K_Corr[:, Diag_nonzeros]/D[Diag_nonzeros].T
    K_Corr = (K_Corr + K_Corr.T)/2

    return 1/N*K_Corr


def Dist2Corr(D):
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
  D = np.sqrt(Diag)[:, np.newaxis]
  K_Corr = np.zeros(K.shape)
  K_Corr[Diag_nonzeros, :] = K[Diag_nonzeros, :]/D[Diag_nonzeros]
  K_Corr[:, Diag_nonzeros] = K_Corr[:, Diag_nonzeros]/D[Diag_nonzeros].T
  K_Corr = (K_Corr + K_Corr.T)/2

  return K_Corr
