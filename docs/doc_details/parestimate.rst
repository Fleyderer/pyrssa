Details
===========

See Sections 3.1 and 5.3 in Golyandina et al (2018) for full details.

Briefly, the time series is assumed to satisfy the model

.. math::
    x_n = \sum_k{C_k\mu_k^n}

for complex :math:`\mu_k` or, alternatively,

.. math::
    x_n = \sum_k{A_k \rho_k^n \sin(2\pi\omega_k n + \phi_k)}.

The return value are the estimated moduli and
arguments of complex :math:`\mu_k`, more precisely, :math:`\rho_k` ('moduli') and :math:`T_k = 1/\omega_k`

For images, the model

.. math::
    x_{ij}=\sum_k C_k \lambda_k^i \mu_k^j

is considered.

References
==============

* Golyandina N., Korobeynikov A., Zhigljavsky A. (2018): Singular Spectrum Analysis with R. Use R!. Springer, Berlin, Heidelberg.

* Roy, R., Kailath, T., (1989): ESPRIT: estimation of signal parameters via rotational invariance techniques. IEEE Trans. Acoust. 37, 984—995.

* Rouquette, S., Najim, M. (2001): Estimation of frequencies and damping factors by two- dimensional esprit type methods. IEEE Transactions on Signal Processing 49(1), 237—245.

* Wang, Y., Chan, J–W., Liu, Zh. (2005): Comments on “estimation of frequencies and damping factors by two-dimensional esprit type methods”. IEEE Transactions on Signal Processing 53(8), 3348—3349.

* Shlemov A, Golyandina N (2014) Shaped extensions of Singular Spectrum Analysis. In: 21st international symposium on mathematical theory of networks and systems, July 7—11, 2014. Groningen, The Netherlands, pp 1813—1820.

See Also
===========

Rssa for an overview of the package, as well as, ssa, lrr,