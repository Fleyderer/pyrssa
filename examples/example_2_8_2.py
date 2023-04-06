import pyrssa as prs
from matplotlib import pyplot as plt

AustralianWine = prs.data("AustralianWine")
Nfull = 120
wine = AustralianWine[:Nfull]
fort_sh = wine["Fortified"].loc["1982-06":"1985-12"]
ss_sh = prs.ssa(fort_sh, L=18)
res_ssa_sh = prs.reconstruct(ss_sh, groups=[1, range(2, 8)])
iss_sh = prs.iossa(ss_sh, nested_groups=[1, range(2, 8)], kappa=0, maxiter=1, tol=1e-5)
res_issa_sh = prs.reconstruct(iss_sh, groups=iss_sh.iossa_groups)
plt.plot(res_ssa_sh.F1, "b-", label="Basic SSA trend", linewidth=0.75)
plt.plot(res_issa_sh.F1, "r-", label="Iterative O-SSA trend", linewidth=1.5)
plt.plot(wine["Fortified"], "k--", label="Full series", linewidth=0.75)
plt.legend()
plt.show()
