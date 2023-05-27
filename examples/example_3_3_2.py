import pyrssa as prs
import numpy as np
from matplotlib import pyplot as plt

co2 = prs.data("co2")
F = co2.copy()
F[200:300] = np.nan
s = prs.ssa(F, L=72)
ig = prs.igapfill(s, groups=[[1, 4]], base="reconstructed")
igo = prs.igapfill(s, groups=[[1, 4]], base="original")
plt.plot(co2, color="black", linewidth=0.75)
plt.plot(ig, color="blue", linewidth=0.75)
plt.plot(igo, color="red", linewidth=0.75)
plt.show()

ig1 = prs.igapfill(s, groups=[[1, 4]], base="original", maxiter=1)
ig5 = prs.igapfill(s, groups=[[1, 4]], fill=ig1, base="original", maxiter=4)
ig10 = prs.igapfill(s, groups=[[1, 4]], fill=ig5, base="original", maxiter=5)
init_lin = F.copy()
init_lin[199:300] = F[199] + np.arange(0, 101) / 101 * (F[300] - F[199])
print(init_lin[197:303])
ig_lin = prs.igapfill(s, fill=init_lin, groups=[[1, 4]], base="original", maxiter=10)

plt.plot(co2, color="black", linewidth=0.75)
plt.plot(ig1, color="lightgreen", linewidth=0.75)
plt.plot(ig5, color="blue", linewidth=0.75)
plt.plot(ig10, color="red", linewidth=0.75)
plt.plot(ig_lin, color="darkred", linewidth=0.75)
plt.show()