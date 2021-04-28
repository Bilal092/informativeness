Python Implementations of Bures and Cosine Informativeness from:

https://jmlr.org/papers/v18/16-296.html


So far the repository contains three modules:
1. Info.py computes Bures and Cosine Informativeness.
2. PSD_Distances.py computes the distances between correlation matrices and it can be easily modified for more general PSD matrices also.
3. PSD_funcs.py contains two functions:
   i) Function PSD2normedCorr computes trace normalized correlation matrix from a PSD Matrix. Trace Normlized Correlation matrices are real valued density matrices.
   ii) Function Dist2Corr computes correlation matrix from a distance matrix.
