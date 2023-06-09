# pyrssa: Python package for Singular Spectrum Analysis

This package contains methods and tools for Singular Spectrum Analysis including decomposition, 
forecasting and gap-filling for univariate and multivariate time series. 
General description of the methods with many examples can be found in the book
Golyandina (2018, <[doi:10.1007/978-3-662-57380-8](https://doi.org/10.1007%2F978-3-662-57380-8)>).

pyrssa is a wrapper over [Rssa package](https://github.com/asl/rssa) made for R language, having its own visualization methods and data structures.

## Install

1. To install the package, make sure you have any [R compiler](https://cran.r-project.org/bin/) installed.
2. Package installer for the latest released version is available at the [Python Package Index (PyPI)](https://pypi.org/project/pyrssa/).
   To install the package, just run the following command:
   
```sh
pip install pyrssa
```

Install requires rpy2, numpy, pandas and matplotlib packages of any version (preferable newest ones).

## Documentation

Documentation for pyrssa package is available [here](https://fleyderer.github.io/pyrssa/).

## Examples

Package usage examples are available in [examples folder](https://github.com/Fleyderer/pyrssa/tree/master/examples).

Original examples with identical numbering are available [here](https://ssa-with-r-book.github.io/).

## Source code

Source code of pyrssa package is available in [pyrssa folder](https://github.com/Fleyderer/pyrssa/tree/master/pyrssa).