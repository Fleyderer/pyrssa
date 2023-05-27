import pandas as pd
import numpy as np
from typing import Callable
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from functools import cached_property
from typing import Literal

r_ssa = rpackages.importr('Rssa')
ssa_get = robjects.r('utils::getFromNamespace("$.ssa", "Rssa")')


class SSABase:

    def __init__(self, x, ssa_object, call: str):
        self.obj = ssa_object
        self._x = x
        self._call = call

    @cached_property
    def sigma(self):
        return np.asarray(ssa_get(self.obj, "sigma"))

    @cached_property
    def U(self):
        return np.asarray(ssa_get(self.obj, "U")).T

    @cached_property
    def V(self):
        return np.asarray(ssa_get(self.obj, "V"))

    @cached_property
    def series(self):
        return pd.Series(self._x)

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
        final_call = self._call[:self._call.find('(') + 1] + "x=" + self._call[self._call.find('(') + 1:]

        result = str(self.obj).split("\n")
        result[result.index("Call:") + 1] = final_call
        return "\n".join(result)

    def __repr__(self):
        return self.__str__()


class SSA(SSABase):

    def __init__(self, x, L: int = None, neig: int = None, mask=None, wmask=None,
                 kind: Literal["1d-ssa", "2d-ssa", "nd-ssa", "toeplitz-ssa", "mssa", "cssa"] = "1d-ssa", circular=False,
                 svd_method: Literal["auto", "nutrlan", "propack", "svd", "eigen", "rspectra", "primme"] = "auto",
                 column_projector="none", row_projector="none", column_oblique="identity",
                 row_oblique="identity", force_decompose: bool = True, call: str = None, **kwargs):

        if L is None:
            L = (len(x) + 1) // 2

        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]

        if row_oblique is not None and not (isinstance(row_oblique, str) and row_oblique == "identity"):
            row_oblique = robjects.FloatVector(row_oblique)
        if column_oblique is not None and not (isinstance(column_oblique, str) and column_oblique == "identity"):
            column_oblique = robjects.FloatVector(column_oblique)

        self.L = L
        self.kind = kind

        super().__init__(x, r_ssa.ssa(x, L=L, neig=neig, mask=mask, wmask=wmask, **kwargs, kind=kind, circular=circular,
                                      **{"svd.method": svd_method, "column.projector": column_projector,
                                         "row.projector": row_projector, "column.oblique": column_oblique,
                                         "row.oblique": row_oblique, "force.decompose": force_decompose}),
                         call=call)


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
        super().__init__(x.series, r_ssa.iossa(x=x, **{"nested.groups": self.nested_groups}, tol=self.tol,
                                               kappa=self.kappa, maxiter=self.maxiter, norm=self.norm,
                                               trace=self.trace,
                                               **{"kappa.balance": self.kappa_balance}, **kwargs), call=call)

    @property
    def iossa_groups(self):
        return self.groups

    def summary(self):
        result = str(self.obj).split("\n")
        result[result.index("Call:") + 1] = self._call
        return "\n".join(result)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


class FOSSA(SSABase):

    def __init__(self, x: SSA,
                 nested_groups,
                 filter=(-1, 1),
                 gamma=np.inf,
                 normalize=True,
                 call=None,
                 **kwargs):
        self.nested_groups = nested_groups
        self.filter = filter
        self.gamma = gamma
        self.normalize = normalize
        super().__init__(x.series, r_ssa.fossa(x=x, **{"nested.groups": self.nested_groups}, filter=self.filter,
                                               gamma=self.gamma, normalize=self.normalize, **kwargs), call=call)

    def summary(self):
        result = str(self.obj).split("\n")
        result[result.index("Call:") + 1] = self._call
        return "\n".join(result)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
