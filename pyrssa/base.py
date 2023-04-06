from pyrssa.classes.SSA import SSABase
from pyrssa.classes.Parestimate import BaseParestimate
from pyrssa.classes.Periodogram import Periodogram
from pyrssa import SSA, IOSSA, FOSSA, Parestimate
from pyrssa import Reconstruction
from pyrssa import RForecast, VForecast, BForecast
from pyrssa import WCorMatrix, HMatrix
from pyrssa import GroupPgram, GroupWCor
from pyrssa import installer
from pyrssa.conversion import pyrssa_conversion_rules
from rpy2 import robjects
import rpy2.robjects.conversion as conversion
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib import callbacks
import pandas as pd
import numpy as np
import os
from typing import overload, Literal, Union
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


# Read pyrssa dataframes
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


def calc_v(x: SSABase, idx, **kwargs):
    return np.asarray(r_ssa.calc_v(x=x, idx=idx, **kwargs)).T


def parestimate(x, groups, method="esprit", subspace="column", normalize_roots=None, dimensions=None,
                solve_method="ls", drop=True):
    """
    Function to estimate the parameters (frequencies and rates) given a set of SSA eigenvectors.

    :param x: SSA object
    :param groups: list of indices of eigenvectors to estimate from
    :param method: For 1D-SSA,
        Toeplitz SSA, and MSSA: parameter estimation method, 'esprit' for 1D-ESPRIT (Algorithm 3.3 in Golyandina et al.
        (2018)), 'pairs' for rough estimation based on pair of eigenvectors (Algorithm 3.4 in Golyandina et al (2018)).
        For nD-SSA: parameter estimation method. For now only 'esprit' is supported (Algorithm 5.6 in Golyandina et al.
        (2018)). lowest dimension, when possible (length of groups is one)
    :param subspace: which subspace will be used for parameter estimation
    :param normalize_roots: logical vector or None, force signal roots to lie on unit circle.
        None means automatic selection: normalize iff circular topology OR Toeplitz SSA used
    :param dimensions: a vector of dimension indices to perform ESPRIT along. None means all dimensions.
    :param solve_method: approximate matrix equation solving method, 'ls' for least-squares,
        'tls' for total-least-squares.
    :param drop: logical, if 'TRUE' then the result is coerced to the lowest dimension,
        when possible (length of groups is one)

    :return:

    """
    if len(groups) == 1:
        return BaseParestimate(x=x, groups=groups, method=method, subspace=subspace,
                               normalize_roots=normalize_roots, dimensions=dimensions,
                               solve_method=solve_method, drop=drop)
    else:
        return Parestimate(x=x, groups=groups, method=method, subspace=subspace,
                           normalize_roots=normalize_roots, dimensions=dimensions,
                           solve_method=solve_method, drop=drop)


def ssa(x, L=None, neig=None, mask=None, wmask=None, kind="1d-ssa", circular=False,
        column_projector="none", row_projector="none", svd_method="auto"):
    """

    :param x: object to be decomposed. If DataFrame passed, the first column will be treated as a series
    :type x: pandas.DataFrame, pandas.Series, numpy.ndarray, list
    :param L: window length. Fixed to half of the series length by default.
        Should be vector of length 2 for 2d SSA
    :type L: int, optional
    :param neig: number of desired eigentriples. If None, then sane default value
        will be used.
    :type neig: int, optional
    :param mask: for shaped 2d SSA case only. Logical matrix with same dimension as x.
        Specifies form of decomposed array. If None, then all non-NA elements will be used
    :param wmask: for shaped 2d SSA case only. Logical matrix which specifies window form.
    :param kind: SSA method. This includes ordinary 1d SSA, 2d SSA, Toeplitz variant of 1d SSA, multichannel
        variant of SSA and complex SSA. Defaults to 1d SSA.
    :type kind: str, optional
    :param circular: logical vector of one or two elements, describes series topology for 1d SSA and Toeplitz SSA
        or field topology for 2d SSA. 'TRUE' means series circularity for 1d case or circularity
        by a corresponding coordinate for 2d case. See Shlemov and Golyandina (2014) for more information.
    :param column_projector, row_projector: column and row signal subspaces projectors for SSA with projection.
    :type column_projector: str or int, optional
    :type row_projector: str or int, optional
    :param svd_method: 	singular value decomposition method.
    :return: SSA object. The precise layout of the object is mostly meant opaque and subject to
        change in different version of the package.
    :rtype: SSA

    Description
    ===========

    Set up the SSA object and perform the decomposition, if necessary.

    Details
    ===========

    This is the main entry point to the package. This routine constructs the SSA object filling all necessary
    internal structures and performing the decomposition if necessary. For the comprehensive description of SSA
    modifications and their algorithms see Golyandina et al. (2018).

    Variants of SSA
    ---------------

    The following implementations of the SSA method are supported (corresponds to different values of kind argument):

    * 1d-ssa

        Basic 1d SSA as described in Chapter 1 of Golyandina et al. (2001). This is also known as Broomhead-King
        variant of SSA or BK-SSA, see Broomhead and King (1986).

    * toeplitz-ssa

        Toeplitz variant of 1d SSA. See Section 1.7.2 in Golyandina et al. (2001). This is also known
        as Vautard-Ghil variant of SSA or VG-SSA for analysis of stationary time series, see Vautard and Ghil (1989).

    * mssa

        Multichannel SSA for simultaneous decomposition of several time series (possible of unequal length). See
        Golyandina and Stepanov (2005).

    * cssa

        Complex variant of 1d SSA.

    * 2d-ssa

        2d SSA for decomposition of images and arrays. See Golyandina and Usevich (2009) and Golyandina et al. (2015)
        for more information.

    * nd-ssa

        Multidimensional SSA decomposition for arrays (tensors).

    Window shape selection (for shaped 2d SSA)
    ------------------------------------------

    Window shape may be specified by argument wmask. If wmask is 'NULL', then standard rectangular window (specified
    by L) will be used.

    * circle(R)

        circular mask of radius R

    * triangle(side)

        mask in form of isosceles right-angled triangle with cathetus side. Right angle lay on top left corner of
        container square matrix

    Also in wmask one may use following functions:

    These functions are not exported, they defined only for wmask expression. If one has objects with the same names
    and wants to use them rather than these functions, one should use special wrapper function I() (see 'Examples').

    Projectors specification for SSA with projection
    ------------------------------------------------

    Projectors are specified by means of column.projector and row.projector arguments (see Golyandina and Shlemov (
    2017)). Each may be a matrix of orthonormal (otherwise QR orthonormalization process will be performed) basis of
    projection subspace, or single integer, which will be interpreted as dimension of orthogonal polynomial basis (
    note that the dimension equals to degree plus 1, e.g. quadratic basis has dimension 3), or one of following
    character strings (or unique prefix): 'none', 'constant' (or 'centering'), 'linear', 'quadratic' or 'qubic' for
    orthonormal bases of the corresponding functions.

    Here is the list of the most used options

    * both projectors are 'none'

        corresponds to ordinary 1D SSA,

    * column.projector='centering'

        corresponds to 1D SSA with centering,

    * column.projector='centering' and row.projector='centering'

        corresponds to 1D SSA with double centering.

    SSA with centering and double centering may improve the separation of linear trend (see Golyandina et al. (2001)
    for more information).

    SVD methods
    -----------

    The main step of the SSA method is the singular decomposition of the so-called series trajectory matrix. Package
    provides several implementations of this procedure (corresponds to different values of svd.method) argument:

    * auto

        Automatic method selection depending on the series length, window length, SSA kind and number of eigenvalues
        requested.

    * nutrlan

        Thick-restart Lanczos eigensolver which operates on cross-product matrix. These methods exploit the Hankel
        structure of the trajectory matrix efficiently and is really fast. The method allows the truncated SVD (only
        specified amount of eigentriples to be computed) and the continuation of the decomposition. See Korobeynikov (
        2010) for more information.

    * propack

        SVD via implicitly restarted Lanczos bidiagonalization with partial reorthogonalization. These methods exploit
        the Hankel structure of the trajectory matrix efficiently and is really fast. This is the 'proper' SVD
        implementation (the matrix of factor vectors are calculated), thus the memory requirements of the methods are
        higher than for nu-TRLAN. Usually the method is slightly faster that nu-TRLAN and more numerically stable.
        The method allows the truncated SVD (only specified amount of eigentriples to be computed). See Korobeynikov (
        2010) for more information.

    * svd

        Full SVD as provided by LAPACK DGESDD routine. Neither continuation of the decomposition nor the truncated
        SVD is supported. The method does not assume anything special about the trajectory matrix and thus is slow.

    * eigen

        Full SVD via eigendecompsition of the cross-product matrix. In many cases faster than previous method,
        but still really slow for more or less non-trivial matrix sizes.

    * rspectra

        SVD via svds function from Rspectra package (if installed)

    * primme

        SVD via svds function from PRIMME package (if installed)

    Usually the ssa function tries to provide the best SVD implementation for given series length and the window
    size. In particular, for small series and window sizes it is better to use generic black-box routines (as
    provided by 'svd' and 'eigen' methods). For long series special-purpose routines are to be used.

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
    return SSA(x, L=L, neig=neig, mask=mask, wmask=wmask, kind=kind,
               circular=circular,
               column_projector=column_projector,
               row_projector=row_projector,
               svd_method=svd_method,
               call=_get_call(inspect.currentframe().f_back))


def reconstruct(x: SSABase,
                groups: Union[list, dict, np.ndarray, GroupPgram, GroupWCor],
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

    By default (argument drop.attributes) the routine tries to preserve all the attributes of the input object. This
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

def rforecast(ds, groups, length=1, base="reconstructed", only_new=True, reverse=False,
              drop=False, drop_attributes=False, cache=True, **kwargs):
    return RForecast(ds, groups, length=length, base=base, only_new=only_new, reverse=reverse,
                     drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def vforecast(ds, groups, length=1, only_new=True, drop=False, drop_attributes=False, **kwargs):
    return VForecast(ds, groups, length=length, only_new=only_new, drop=drop, drop_attributes=drop_attributes, **kwargs)


def bforecast(x, groups, length=1, R=100, level=0.95, kind="recurrent", interval="confidence",
              only_new=True, only_intervals=False, drop=True, drop_attributes=False, cache=True, **kwargs):
    return BForecast(x, groups, length=length, r=R, level=level, kind=kind, interval=interval, only_new=only_new,
                     only_intervals=only_intervals, drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


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


def spectrum(x, spans=None, kernel=None, taper=0.1,
             pad=0, fast=True, demean=False, detrend=True, **kwargs):
    return Periodogram(x, spans=spans, kernel=kernel, taper=taper,
                       pad=pad, fast=fast, demean=demean, detrend=detrend, **kwargs)
