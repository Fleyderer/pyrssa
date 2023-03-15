import pandas as pd
import numpy as np
from typing import Callable
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
import inspect


r_ssa = rpackages.importr('Rssa')
ssa_get = robjects.r('utils::getFromNamespace("$.ssa", "Rssa")')


class SSABase:

    def __init__(self, x, ssa_object, call: str):
        self.obj = ssa_object
        self.sigma = ssa_get(self.obj, "sigma")
        self.U = ssa_get(self.obj, "U").T
        self.V = ssa_get(self.obj, "V")
        self.series = pd.DataFrame(x)
        self.call = call

    @property
    def F(self):
        return self.series

    def contributions(self, idx=None):
        if idx is None:
            idx = range(1, self.nsigma() + 1)
        return r_ssa.contributions(self.obj, idx)

    def nspecial(self):
        return r_ssa.nspecial(self.obj)[0]

    def nsigma(self):
        return r_ssa.nsigma(self.obj)[0]

    def nu(self):
        return r_ssa.nu(self.obj)[0]

    def __str__(self):
        result = str(self.obj).split("\n")
        result[result.index("Call:") + 1] = self.call
        return "\n".join(result)

    def __repr__(self):
        return self.__str__()


class SSA(SSABase):

    def __init__(self, x,
                 L=None,
                 neig=None,
                 mask=None,
                 wmask=None,
                 kind="1d-ssa",
                 circular=False,
                 column_projector="none",
                 row_projector="none",
                 svd_method="auto",
                 call=None):

        if L is None:
            L = (len(x) + 1) // 2

        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]

        self.L = L
        self.kind = kind
        super().__init__(x, r_ssa.ssa(x, L=L, neig=neig, mask=mask, wmask=wmask, kind=kind,
                                      circular=circular,
                                      column_projector=column_projector,
                                      row_projector=row_projector,
                                      svd_method=svd_method), call=call)


def _norm_conversion(func):
    def wrapper(x):
        return float(func(np.array(x)))
    return wrapper


def _default_norm(x):
    return np.sqrt(np.mean(x ** 2))


class IOSSA(SSABase):

    def __init__(self, x: SSA,
                 nested_groups,
                 tol=1e-5,
                 kappa=2,
                 maxiter=100,
                 norm: Callable = _default_norm,
                 trace=False,
                 kappa_balance=0.5,
                 call=None,
                 **kwargs):
        self.nested_groups = nested_groups
        self.tol = tol
        self.kappa = kappa
        self.maxiter = maxiter
        self.kappa_balance = kappa_balance
        self.trace = trace
        self.groups = nested_groups
        if norm is None:
            norm = _default_norm
        self.norm = robjects.rinterface.rternalize(_norm_conversion(norm))
        super().__init__(x.F, r_ssa.iossa(x=x, **{"nested.groups": self.nested_groups}, tol=self.tol,
                                          kappa=self.kappa, maxiter=self.maxiter, norm=self.norm,
                                          trace=self.trace,
                                          **{"kappa.balance": self.kappa_balance}, **kwargs), call=call)

    @property
    def iossa_groups(self):
        return self.groups

    def summary(self):
        result = str(self.obj).split("\n")
        result[result.index("Call:") + 1] = self.call
        return "\n".join(result)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
