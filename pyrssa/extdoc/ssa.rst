Details
-----------

This is the main entry point to the package. This routine constructs the SSA object filling all necessary
internal structures and performing the decomposition if necessary. For the comprehensive description of SSA
modifications and their algorithms see Golyandina et al. (2018).

Variants of SSA
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^

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