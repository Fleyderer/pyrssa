import numpy as np
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from functools import cached_property

r_ssa = rpackages.importr('Rssa')


class BaseParestimate:

    def __init__(self, par_obj=None, x=None, groups=None, method="esprit", subspace="column", normalize_roots=None,
                 dimensions=None, solve_method="ls", drop=True):
        if par_obj is not None:
            self.obj = par_obj
        else:
            self.obj = r_ssa.parestimate(x=x, groups=groups, method=method, subspace=subspace,
                                         normalize_roots=normalize_roots, dimensions=dimensions,
                                         solve_method=solve_method, drop=drop)

    @cached_property
    def roots(self):
        return np.asarray(self.obj.rx("roots")[0])

    @cached_property
    def periods(self):
        return np.asarray(self.obj.rx("periods")[0])

    @cached_property
    def frequencies(self):
        return np.asarray(self.obj.rx("frequencies")[0])

    @cached_property
    def moduli(self):
        return np.asarray(self.obj.rx("moduli")[0])

    @cached_property
    def rates(self):
        return np.asarray(self.obj.rx("rates")[0])

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise AttributeError(f"This parestimate group does not have attribute with name '{item}'.")

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


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
        self.names = list(robjects.r.names(self.obj))

    def __getattribute__(self, item) -> BaseParestimate:
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item in self.names:
                res = BaseParestimate(par_obj=self.obj.rx(item)[0])
                setattr(self, item, res)
                return res
            else:
                raise AttributeError(f"This parestimate object does not have group with name '{item}'.")

    def __getitem__(self, item) -> BaseParestimate:
        if isinstance(item, str) and item in self.keys():
            return getattr(self, item)
        else:
            raise AttributeError(f"This parestimate object does not have group with name '{item}'.")

    def keys(self):
        return self.names

    def values(self):
        return [self.__getitem__(name) for name in self.names]

    def items(self):
        return zip(self.keys(), self.values())

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
