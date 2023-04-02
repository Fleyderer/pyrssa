from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class Parestimate:

    def __init__(self, x, groups, method="esprit", subspace="column", normalize_roots=None,
                 dimensions=None, solve_method="ls", drop=True):

        if type(x) == SSA:
            method = "esprit"
            subspace = "column"
            normalize_roots = None
            dimensions = None
            solve_method = "ls"
            drop = True

        self.obj = r_ssa.parestimate(x=x, groups=groups, method=method, subspace=subspace,
                                     normalize_roots=normalize_roots, dimensions=dimensions,
                                     solve_method=solve_method, drop=drop)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
