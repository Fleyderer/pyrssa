import pyrssa as prs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

co2 = prs.data("co2")
F = co2.copy()
loc = np.concatenate([np.arange(10, 17), np.arange(60, 67),
                      np.arange(70, 77), np.arange(100, 107)])
F[loc] = np.nan
sr = prs.ssa(F, L=200)
igr = prs.igapfill(sr, groups=[range(1, 7)], fill=320, base="original", maxiter=10)
gr = prs.gapfill(sr, groups=[range(1, 7)], method="simultaneous", base="original")
G = pd.Series(np.repeat(np.nan, len(F)), index=gr.index)
G[loc] = gr[loc]

print(np.mean((gr[loc] - co2[loc]) ** 2))  # MSE of gapfill
print(np.mean((igr[loc] - co2[loc]) ** 2))  # MSE of igapfill

plt.plot(igr, linewidth=1, label="igapfill")
plt.plot(G, linewidth=1, label="gapfill")
plt.plot(F, linewidth=1, label="series")
plt.legend()
plt.show()
