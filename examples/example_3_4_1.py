import pyrssa as prs
import numpy as np
from matplotlib import pyplot as plt

co2 = prs.data("co2")
cut = 49 + 60
x = co2[:-cut + 1]
L = 60
K = len(x) - L + 1
alpha = 0.01
weights = np.empty(K)
weights[:K] = alpha
weights[range(K - 1, -1, -L)] = 1
plt.plot(weights)
plt.show()

s1 = prs.ssa(x, L=L)
ncomp = 6
s01 = prs.ssa(x, L=L, column_oblique="identity", row_oblique=weights)
c01 = prs.cadzow(s01, rank=ncomp, maxiter=10)
s01_1 = prs.ssa(c01, L=L, column_oblique=None, row_oblique=None)
c01_1 = prs.cadzow(s01_1, rank=ncomp, tol=1e-8 * np.mean(co2))
ss01_1 = prs.ssa(c01_1, L=ncomp + 1)
fr = prs.rforecast(ss01_1, groups=[range(1, ncomp + 1)], length=cut)
plt.plot(co2, label="Original")
plt.plot(c01_1, label="Cadzow1and01")
plt.plot(fr, label="ForecastCadzow")
plt.show()
print(prs.parestimate(ss01_1, groups=[range(1, ncomp + 1)], method="esprit"))
