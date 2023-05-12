import pyrssa as prs
import numpy as np
from matplotlib import pyplot as plt

co2 = prs.data("co2")
F = co2.copy()
F[200:300] = np.nan
s = prs.ssa(F, L=72)
g0 = prs.gapfill(s, [[1, 4]], method="sequential", alpha=0, base="reconstructed")
g1 = prs.gapfill(s, [[1, 4]], method="sequential", alpha=1, base="reconstructed")
g = prs.gapfill(s, [[1, 4]], method="sequential", base="reconstructed")
plt.plot(co2, color="black", linewidth=0.5)
plt.plot(g0, color="blue")
plt.plot(g1, color="lightgreen")
plt.plot(g, color="red")
plt.show()

