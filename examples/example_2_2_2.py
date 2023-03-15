import pyrssa as prs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

simul = False
n = 100
sigma = 0.5
np.random.seed(8)
alpha = np.arange(0, 0.011, 0.001)
L = 50
Q = 1000

if simul:

    def rmse(a):
        S = np.exp(a * np.arange(1, n + 1)) * np.sin(2 * np.pi * np.arange(1, n + 1) / 7)
        f = S + sigma * np.random.normal(size=n)
        f_center = f - np.mean(f)
        s = prs.ssa(f, L=L, kind="1d-ssa")
        st = prs.ssa(f_center, L=L, kind='toeplitz-ssa')
        rec = prs.reconstruct(s, groups=[[1, 2]]).F1
        rect = prs.reconstruct(st, groups=[[1, 2]]).F1
        return {"1d-ssa": np.mean((rec - S) ** 2),
                "toeplitz": np.mean((rect - S) ** 2)}

    RMSE = [np.sqrt(np.mean(pd.DataFrame.from_records([rmse(a) for _ in range(Q)]), axis=0)) for a in alpha]
    toeplitz_sim = pd.DataFrame(RMSE)
else:
    toeplitz_sim = prs.data("toeplitz.sim")

ax = toeplitz_sim.plot()
ax.set_xticks(toeplitz_sim.index[::2])
ax.set_xticklabels(alpha[::2])

plt.show()
