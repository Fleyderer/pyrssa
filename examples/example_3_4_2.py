import numpy as np
import pandas as pd
import pyrssa as prs

simul = False
np.random.seed(3)
L = 20
N = 2 * L
K = N - L + 1
alpha = 0.01
sigma = 1
signal = 5 * np.sin(2 * np.pi * np.arange(1, N + 1) / 6)
weights = np.empty(K)
weights[:K] = alpha
weights[range(0, K + 1, L)] = 1
M = 1000


def norm_meansq(x):
    return np.mean(x ** 2)


if simul:
    def rmse():
        x = signal + sigma * np.random.normal(size=N)
        s_alpha = prs.ssa(x, L=L, column_oblique=None, row_oblique=weights)
        c_alpha = prs.cadzow(s_alpha, rank=2, tol=1e-8, norm=norm_meansq, correct=False)
        s = prs.ssa(x, L=L)
        cc = prs.cadzow(s, rank=2, norm=norm_meansq, tol=1e-8, correct=False)
        return {"err": np.mean((cc - signal) ** 2),
                "err_alpha": np.mean((c_alpha - signal) ** 2)}


    RMSE = [np.sqrt(np.mean(pd.DataFrame.from_records([rmse() for _ in range(M)]), axis=0))]
    cadzow_sim = pd.DataFrame(RMSE)
    pd.set_option("display.precision", 20)
else:
    cadzow_sim = prs.data("cadzow.sim")

print(cadzow_sim)
