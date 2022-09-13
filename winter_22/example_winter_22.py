from pyrssa import base
import numpy as np
import matplotlib.pyplot as plt

N = 100
sigma = 0.5
noise = np.array(base.r.rnorm(N)) * sigma
trend = 0.65 * base.seq(1, N)
seasonality = np.sin(2 * base.pi * base.seq(1, N) / 6)
series = seasonality + noise + trend
s_series = base.ssa(series, L=21, kind="1d-ssa")
r_series = base.reconstruct(s_series, groups=base.list(Trend=range(1, 3), Seasonality=range(3, 5)))
fig, ax = plt.subplots(2, 2)
fig.suptitle("Reconstruction")
ax[0, 0].plot(base.seq(1, N), series, label="Original")
ax[0, 1].plot(base.seq(1, N), r_series.rx("Trend")[0], label='Trend')
ax[0, 1].plot(base.seq(1, N), trend, label='Original trend')
ax[0, 1].legend()
ax[1, 0].plot(base.seq(1, N), r_series.rx("Seasonality")[0], label='Seasonality')
ax[1, 0].plot(base.seq(1, N), seasonality, label='Original seasonality')
ax[1, 0].legend()
ax[1, 1].plot(base.seq(1, N), base.r.attr(r_series, "residuals"), label='Residuals')
ax[1, 1].plot(base.seq(1, N), noise, label='Original residuals')
ax[1, 1].legend()
plt.show()





