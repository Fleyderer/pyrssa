import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class HMatrix(np.matrix):

    def __new__(cls, F, B=None, T=None, L=None, neig=10):
        n = len(F)
        if B is None:
            B = n // 4
        if T is None:
            T = n // 4
        if L is None:
            L = B // 2
        obj = np.asmatrix(r_ssa.hmatr(F=F, B=B, T=T, L=L, neig=neig).view(cls))
        return obj

    def __array_finalize__(self, obj):
        pass
