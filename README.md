Python Implementations of Bures and Cosine Informativeness from:

https://jmlr.org/papers/v18/16-296.html

The Code in repository is still under development. So far it contains three modules:
1. Info.py contains two function:
     i) Function Info computes Bures and Cosine Informativeness for general correlation operators with diagonals of ones.
     ii) Function Info_embeddings computes Bures Informativeness for embedding Z of correlation matrix A onto oblique manifold, where A = Z*Z.T is correlation matrix.
3. PSD_Distances.py computes the distances between correlation matrices and it can be easily modified for more general PSD matrices also.
4. PSD_funcs.py contains two functions:
   i) Function PSD2normedCorr computes trace normalized correlation matrix from a PSD Matrix. Trace Normlized Correlation matrices are real valued density matrices.
   ii) Function Dist2Corr computes correlation matrix from a distance matrix.
