import numpy as np
import pandas as pd
import pyrssa as prs
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def poly(x, d):
    x = np.array(x, dtype=np.int64)
    return np.transpose([x ** k for k in range(1, d + 1)])


N = 199
tt = np.arange(1, N + 1) / N
r = 5
F0 = 10 * (tt - 0.5) ** r
F = F0 + np.sin(2 * np.pi * np.arange(1, N + 1) / 10)
L = 100
dec = prs.ssa(F, L=L, column_projector=3, row_projector=3)
rec1 = prs.reconstruct(dec, groups={"Trend": np.arange(1, dec.nspecial() + 1)})
fit1 = rec1.Trend
p = poly(np.arange(1, N + 1), 5)
fit1_3b = LinearRegression()
fit1_3b.fit(p, fit1)
fit3b = LinearRegression()
fit3b.fit(p, F)
li = np.arange(199)
df = pd.DataFrame({"Initial": F[li],
                   "dproj": fit1[li],
                   "dproj_reg": fit1_3b.predict(p)[li],
                   "regr": fit3b.predict(p)[li],
                   "trend": F0[li]})
df.plot(y=df.columns)
plt.show()
