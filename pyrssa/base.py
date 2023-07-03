from pyrssa.classes.SSA import SSABase
from pyrssa.classes.Parestimate import BaseParestimate
from pyrssa.classes.LRR import BaseLRR
from pyrssa.classes.Periodogram import Periodogram
from pyrssa.docs import add_doc
from pyrssa import SSA, IOSSA, FOSSA, Parestimate, LRR
from pyrssa import Cadzow
from pyrssa import Reconstruction
from pyrssa import RForecast, VForecast, BForecast, Forecast
from pyrssa import Gapfill, IGapfill
from pyrssa import WCorMatrix, HMatrix
from pyrssa import GroupPgram, GroupWCor
from pyrssa import installer
from pyrssa.conversion import pyrssa_conversion_rules
from rpy2 import robjects
import rpy2.robjects.conversion as conversion
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib import callbacks
import pandas as pd
import numpy
import numpy as np
import os
from typing import overload, Literal, Union, Callable
from collections.abc import Iterable
import inspect

# Set conversion rules
conversion.set_conversion(pyrssa_conversion_rules)

# Ignore warnings
callbacks.consolewrite_warnerror = lambda *args: None

# Install required R packages.
installer.install_required()

r = robjects.r
r_ssa = rpackages.importr('Rssa')


def _get_call(frame):
    call = inspect.getframeinfo(frame)[3][0]
    return call[call.find('=') + 1:].strip().rstrip()


def _set_datetime_index(dataframe, name):
    dataframe[name] = pd.DatetimeIndex(dataframe[name])
    dataframe.set_index(name, inplace=True, drop=True)
    dataframe.index.freq = dataframe.index.inferred_freq


def data(name, datetime_index=None):
    """
    Function for loading available in pyrssa package datasets. Available datasets are stored in the data directory.

    :param name: Name of dataset to load
    :param datetime_index: Name of column, where date or time index is stored
    :return: Loaded dataset
    :rtype: pandas.DataFrame or pandas.Series

    """
    result = pd.read_csv(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), f'{name}.csv'))
    if datetime_index is not None:
        _set_datetime_index(result, datetime_index)
    elif len(result.columns) > 1:
        # Search for datetime index and set it
        col_idx = next((i for i, v in enumerate(result.columns) if v.lower() in ['time', 'date', 'datetime']), None)
        if col_idx is not None:
            _set_datetime_index(result, result.columns[col_idx])

    # If there is only one column in dataframe, return it as series
    if len(result.columns) == 1:
        return result[result.columns[0]]
    else:
        return result


@add_doc
def calc_v(x: SSABase, idx: Union[int, list, range, numpy.ndarray], **kwargs) -> numpy.ndarray:
    """
    Generic function for the factor vector calculation given the SSA decomposition.

    Parameters
    ----------
    x
        SSA object holding the decomposition.
    idx
        indices of the factor vectors to compute.
    kwargs
        additional arguments.

    Returns
    -------
        numpy array of suitable length (usually depends on SSA method and window length).


    Examples
    --------

    .. code-block:: python

        co2 = prs.data("co2")
        # Decompose 'co2' series with default parameters
        s = prs.ssa(co2)
        # Calculate the 5th factor vector
        v = prs.calc_v(s, 5)
    """
    return np.asarray(r_ssa.calc_v(x=x, idx=idx, **kwargs)).T


@add_doc
def parestimate(x: SSA, groups: Iterable, method: Literal["esprit", "pairs"] = "esprit",
                dimensions: list = None, subspace: Literal["column", "row"] = "column",
                normalize_roots: Union[bool, list] = None, solve_method: Literal["ls", "tls"] = "ls",
                drop: bool = True, **kwargs) -> BaseParestimate | Parestimate:
    r"""
    Function to estimate the parameters (frequencies and rates) given a set of SSA eigenvectors.

    Parameters
    ----------
    x:
        SSA object
    groups:
        list of indices of eigenvectors to estimate from
    method:
        for 1D-SSA, Toeplitz SSA, and MSSA: parameter estimation method, 'esprit' for 1D-ESPRIT (Algorithm 3.3 in
        Golyandina et al. (2018)), 'pairs' for rough estimation based on pair of eigenvectors (Algorithm 3.4 in
        Golyandina et al (2018)). For nD-SSA: parameter estimation method. For now only 'esprit' is supported
        (Algorithm 5.6 in Golyandina et al. (2018)). lowest dimension, when possible (length of groups is one)
    dimensions:
        a vector of dimension indices to perform ESPRIT along. None means all dimensions.
    subspace:
        which subspace will be used for parameter estimation
    normalize_roots:
        force signal roots to lie on unit circle.
        None means automatic selection: normalize iff circular topology OR Toeplitz SSA used
    solve_method:
        approximate matrix equation solving method, 'ls' for least-squares,
        'tls' for total-least-squares.
    drop:
        if True then the result is coerced to the lowest dimension,
        when possible (length of groups is one).
    **kwargs:
        further arguments passed to 'decompose' routine, if necessary

    Returns
    -------

        |Parestimate| For 1D-SSA (and Toeplitz) - a list of |BaseParestimate| objects. For method = 'pairs'
        all moduli are set equal to 1 and all rates equal to 0. In all cases elements of the list have the same names
        as elements of groups. If group is unnamed, corresponding component gets name ‘Fn’, where ‘n’ is its index in
        groups list. If drop is True and length of groups is one, then |BaseParestimate| is returned.

    Examples
    --------

    .. code-block:: python

        co2 = prs.data("co2")
        # Decompose 'co2' series with default parameters
        s = prs.ssa(co2, neig=20)
        # Estimate the periods from 2nd and 3rd eigenvectors using 'pairs' method
        print(prs.parestimate(s, groups=[[2, 3]], method = "pairs"))
        # Estimate the periods from 2nd, 3rd, 5th and 6th eigenvectors using ESPRIT
        pe = prs.parestimate(s, groups=[[2, 3, 5, 6]], method="esprit")
        print(pe)
        prs.plot(pe)
    """
    return Parestimate(x=x, groups=groups, method=method, dimensions=dimensions,
                       subspace=subspace, normalize_roots=normalize_roots,
                       solve_method=solve_method, drop=drop, **kwargs)


@add_doc
def ssa(x: Iterable, L: int = None, neig: int = None, mask=None, wmask=None,
        kind: Literal["1d-ssa", "2d-ssa", "nd-ssa", "toeplitz-ssa", "mssa", "cssa"] = "1d-ssa", circular=False,
        svd_method: Literal["auto", "nutrlan", "propack", "svd", "eigen", "rspectra", "primme"] = "auto",
        column_projector="none", row_projector="none", column_oblique="identity",
        row_oblique="identity",  force_decompose: bool = True, **kwargs) -> SSA:
    """
    Set up the SSA object and perform the decomposition, if necessary.

    Parameters
    ----------
    x:
        object to be decomposed. If DataFrame passed, the first column will be treated as a series.
    L:
        window length. Fixed to half of the series length by default. Should be a vector of length 2 for 2d SSA.
    neig:
        number of desired eigentriples. If None, then sane default value will be used.
    mask:
        for shaped 2d SSA case only. Logical matrix with same dimension as x. Specifies form of decomposed array.
         If None, then all non-NA elements will be used
    wmask:
        for shaped 2d SSA case only. Logical matrix which specifies window form.
    kind:
        SSA method. This includes ordinary 1d SSA, 2d SSA, Toeplitz variant of 1d SSA, multichannel
        variant of SSA and complex SSA. Defaults to 1d SSA.
    circular:
        logical vector of one or two elements, describes series topology for 1d SSA and Toeplitz SSA
        or field topology for 2d SSA. 'TRUE' means series circularity for 1d case or circularity
        by a corresponding coordinate for 2d case. See Shlemov and Golyandina (2014) for more information.
    column_projector, row_projector:
        column and row signal subspaces projectors for SSA with projection.
    svd_method:
        singular value decomposition method.

    Returns
    -------

        SSA object. The precise layout of the object is mostly meant opaque and subject to
        change in different version of the package.

    Examples
    --------

    .. code-block:: python

        import pyrssa as prs
        import pandas as pd
        import numpy as np

        AustralianWine = prs.data("AustralianWine")
        fort = AustralianWine['Fortified'][:174]
        fort.index = pd.date_range(start='1980/01/01', freq='M', periods=len(fort))
        s_fort = prs.ssa(fort, L=84, kind="1d-ssa")

        co2 = data("co2")
        s2 = prs.ssa(co2.value, column_projector="centering", row_projector="centering")
        s4 = prs.ssa(co2.value, column_projector=2, row_projector=2)

        S = np.exp(a * np.arange(1, n + 1)) * np.sin(2 * np.pi * np.arange(1, n + 1) / 7)
        f = S + sigma * np.random.normal(size=n)
        f_center = f - np.mean(f)
        s = prs.ssa(f, L=L, kind="1d-ssa")
        st = prs.ssa(f_center, L=L, kind='toeplitz-ssa')

    """
    return SSA(x, L=L, neig=neig, mask=mask, wmask=wmask, kind=kind, circular=circular,
               svd_method=svd_method, column_projector=column_projector, row_projector=row_projector,
               column_oblique=column_oblique, row_oblique=row_oblique, force_decompose=force_decompose,
               call=_get_call(inspect.currentframe().f_back), **kwargs)


def reconstruct(x: SSABase,
                groups: Iterable,
                drop_attributes=False,
                cache=True):
    """

    :param x: SSA object
    :type x: SSA
    :param groups: list of numeric vectors, indices of elementary components used for reconstruction,
        the entries of the list can be named, see 'Value' for more information
    :type groups: list or dict
    :param drop_attributes: if `True` then the attributes of the input objects are not copied to the reconstructed ones.
    :type drop_attributes: bool
    :param cache: if `True` then intermediate results will be cached in the SSA object.
    :type cache: bool
    :return: List of reconstructed objects. Elements of the list have the same names as elements of groups. If the
        group is unnamed, then corresponding component will obtain name ‘Fn’, where ‘n’ is its index in groups list.
    :rtype: Reconstruction

    Description
    ===========

    Reconstruct the data given the SSA decomposition and the desired grouping of the elementary components.

    Details
    =======

    Reconstruction is performed in a common form for different types of input objects. See Section 1.1.2.6 in
    Golyandina et al. (2018) for the explanation. Formal algorithms are described in this book in Algorithm 2.2 for
    1D-SSA, Algorithm 4.3 for MSSA, Algorithm 5.2 for 2D-SSA and Algorithm 5.6 for Shaped 2D-SSA.

    Fast implementation of reconstruction with the help of FFT is described in Korobeynikov (2010) for the 1D case
    and in Section 6.2 (Rank-one quasi-hankelization) of Golyandina et al. (2015) for the general case.


    Note
    ----

    By default (argument drop_attributes) the routine tries to preserve all the attributes of the input object. This
    way, for example, the reconstruction result of 'ts' object is the 'ts' object with the same time scale.

    References
    ----------

    Golyandina N., Korobeynikov A., Zhigljavsky A. (2018): Singular Spectrum Analysis with R. Use R!. Springer,
    Berlin, Heidelberg.

    Korobeynikov, A. (2010): Computation- and space-efficient implementation of SSA. Statistics and Its Interface,
    Vol. 3, No. 3, Pp. 257-268

    Golyandina, N., Korobeynikov, A., Shlemov, A. and Usevich, K. (2015): Multivariate and 2D Extensions of Singular
    Spectrum Analysis with the Rssa Package. Journal of Statistical Software, Vol. 67, Issue 2.
    doi:10.18637/jss.v067.i02

    Examples
    --------

    .. code-block:: python

        import pyrssa as prs
        import pandas as pd
        import numpy as np

        AustralianWine = prs.data("AustralianWine")
        fort = AustralianWine['Fortified'][:174]
        fort.index = pd.date_range(start='1980/01/01', freq='M', periods=len(fort))
        s_fort = prs.ssa(fort, L=84, kind="1d-ssa")
        r_fort = prs.reconstruct(s_fort, groups={"Trend": 1, "Seasonality": range(2, 13)})

        r_fort_2 = prs.reconstruct(s_fort, groups=[1, range(2, 13)])

        co2 = data("co2")
        s2 = prs.ssa(co2.value, column_projector="centering", row_projector="centering")
        rec = prs.reconstruct(s2, groups={"Linear_trend": range(1, s2.nspecial() + 1)})

    """
    return Reconstruction(x=x, groups=groups, drop_attributes=drop_attributes, cache=cache)


def cadzow(x: SSA, rank: int, correct: bool = True, tol: float = 1e-6, maxiter: int = 0,
           norm: Callable = None, trace: bool = False, cache: bool = True, **kwargs):

    return Cadzow(x=x, rank=rank, correct=correct, tol=tol, maxiter=maxiter,
                  norm=norm, trace=trace, cache=cache, **kwargs)


def iossa(x: SSA, nested_groups, tol=1e-5, kappa=2, maxiter=100, norm=None, trace=False, kappa_balance=0.5, **kwargs):
    return IOSSA(x=x, nested_groups=nested_groups, tol=tol, kappa=kappa, maxiter=maxiter, norm=norm, trace=trace,
                 kappa_balance=kappa_balance, call=_get_call(inspect.currentframe().f_back), **kwargs)


def fossa(x: SSA, nested_groups, filter=(-1, 1), gamma=np.inf, normalize=True, call=None, **kwargs):
    return FOSSA(x=x, nested_groups=nested_groups, filter=filter,
                 gamma=gamma, normalize=normalize, **kwargs, call=call)


# Weighted correlations
def wcor(ds, groups=range(1, 51)):
    return WCorMatrix(ds, groups)


# Forecasting functions

def rforecast(x, groups, length=1, base="reconstructed", only_new=True, reverse=False,
              drop=True, drop_attributes=False, cache=True, **kwargs):
    return RForecast(x=x, groups=groups, length=length, base=base, only_new=only_new, reverse=reverse,
                     drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def vforecast(x, groups, length=1, only_new=True, drop=True, drop_attributes=False, **kwargs):
    return VForecast(x=x, groups=groups, length=length, only_new=only_new,
                     drop=drop, drop_attributes=drop_attributes, **kwargs)


def bforecast(x, groups, length=1, R=100, level=0.95, kind="recurrent", interval="confidence",
              only_new=True, only_intervals=False, drop=True, drop_attributes=False, cache=True, **kwargs):
    return BForecast(x, groups, length=length, r=R, level=level, kind=kind, interval=interval, only_new=only_new,
                     only_intervals=only_intervals, drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def forecast(x: SSA, groups, length=1, method: Literal["recurrent", "vector"] = "recurrent",
             interval: Literal["none", "confidence", "prediction"] = "none",
             only_intervals=True, direction: Literal["column", "row"] = "column",
             drop=True, drop_attributes=False, cache=True, seed: int = None, **kwargs):
    return Forecast(x=x, groups=groups, length=length, method=method, interval=interval,
                    only_intervals=only_intervals, direction=direction, drop=drop,
                    drop_attributes=drop_attributes, cache=cache, seed=seed, **kwargs)


def gapfill(x: SSA, groups, base: Literal["original", "reconstructed"] = "original",
            method: Literal["sequential", "simultaneous"] = "sequential", alpha: Union[int, float, Callable] = None,
            drop=True, drop_attributes=False, cache=True, **kwargs):
    return Gapfill(x=x, groups=groups, base=base, method=method, alpha=alpha,
                   drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def igapfill(x: SSA, groups, fill=None, tol: float = 1e-6, maxiter=0, norm=None,
             base: Literal["original", "reconstructed"] = "original", trace=False,
             drop=True, drop_attributes=False, cache=True, **kwargs):
    return IGapfill(x=x, groups=groups, fill=fill, tol=tol, maxiter=maxiter, norm=norm, base=base,
                   trace=trace, drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def hmatr(F, B=None, T=None, L=None, neig=10):
    return HMatrix(F, B=B, T=T, L=L, neig=neig)


def grouping_auto_pgram(x: SSABase, groups=None, base: Literal["series", "eigen", "factor"] = "series",
                        freq_bins=2, threshold=0, method: Literal["constant", "linear"] = "constant",
                        drop=True, **kwargs):
    return GroupPgram(x=x, groups=groups, base=base, freq_bins=freq_bins, threshold=threshold,
                      method=method, drop=drop, **kwargs)


def grouping_auto_wcor(x: SSABase, groups=None, nclust=None, **kwargs):
    return GroupWCor(x=x, groups=groups, nclust=nclust, **kwargs)


@overload
def grouping_auto(x: SSABase, grouping_method: Literal["pgram"] = "pgram", groups=None, nclust=None,
                  base: Literal["series", "eigen", "factor"] = "series", freq_bins=2, threshold=0,
                  method: Literal["constant", "linear"] = "constant", drop=True, **kwargs) -> GroupPgram:
    ...


@overload
def grouping_auto(x: SSABase, grouping_method: Literal["wcor"] = "wcor", groups=None, nclust=None,
                  base: Literal["series", "eigen", "factor"] = "series",
                  freq_bins=2, threshold=0,
                  method: Literal["ward.D", "ward.D2", "single", "complete",
                  "average", "mcquitty", "median", "centroid"] = "complete",
                  drop=True, **kwargs) -> GroupWCor:
    ...


def grouping_auto(x: SSABase, grouping_method: str = "pgram", groups=None, nclust=None,
                  base: Literal["series", "eigen", "factor"] = "series", freq_bins=2, threshold=0,
                  method: Literal["constant", "linear"] = "constant", drop=True, **kwargs):
    if grouping_method == "pgram":
        return grouping_auto_pgram(x, groups=groups, base=base, freq_bins=freq_bins, threshold=threshold,
                                   method=method, drop=drop, **kwargs)
    elif grouping_method == "wcor":
        return grouping_auto_wcor(x, groups=groups, nclust=nclust, method=method, **kwargs)
    else:
        raise ValueError(f"Grouping method {grouping_method} is not in available methods: 'pgram', 'wcor'")


@overload
def lrr(x: SSABase, groups=None, reverse=False, drop=True):
    if len(groups) == 1 and drop:
        return BaseLRR(x=x, groups=groups, reverse=reverse, drop=drop)
    else:
        return LRR(x=x, groups=groups, reverse=reverse, drop=drop)


def lrr(x, groups=None, reverse=False, drop=True, eps=np.sqrt(np.finfo(float).eps),
        raw=False, orthonormalize=True):
    if len(groups) == 1 and drop:
        return BaseLRR(x=x, groups=groups, reverse=reverse, drop=drop,
                       eps=eps, raw=raw, orthonormalize=orthonormalize)
    else:
        return LRR(x=x, groups=groups, reverse=reverse, drop=drop,
                   eps=eps, raw=raw, orthonormalize=orthonormalize)


def spectrum(x, spans=None, kernel=None, taper=0.1,
             pad=0, fast=True, demean=False, detrend=True, **kwargs):
    return Periodogram(x, spans=spans, kernel=kernel, taper=taper,
                       pad=pad, fast=fast, demean=demean, detrend=detrend, **kwargs)
