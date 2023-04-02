import numpy as np
import pyrssa as prs
from matplotlib import pyplot as plt


def row_means_quantile(x, level=0.05):
    x = np.array([np.fromstring(row[1:-1], sep=',') for row in x.values])

    def quantile(x):
        x = np.array(x, dtype=float)
        q = np.percentile(x, np.array([level / 2, 1 - level / 2]) * 100)
        x[x < q[0]] = q[0]
        x[x > q[1]] = q[1]
        return np.mean(x)
    return np.apply_along_axis(quantile, 1, x)


iossa_err = prs.data("iossa.err")
lseq = np.concatenate([np.arange(0.03, 0.058, 0.002), np.arange(0.062, 0.1, 0.002)])
iter_real = row_means_quantile(iossa_err["iter.real"])
iter_est = iossa_err["iter.est"]
err1 = np.sqrt(row_means_quantile(iossa_err["err1"]))
err2 = np.sqrt(row_means_quantile(iossa_err["err2"]))

plt.figure(figsize=(12, 4))
plt.plot(lseq, iter_real)
plt.plot(lseq, iter_est)
plt.xlabel(r"$\omega_1$")
plt.ylabel(r"$N_{iter}$")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(lseq, err1)
plt.plot(lseq, err2)
plt.xlabel(r"$\omega_1$")
plt.ylabel(r"$RMSE$")
plt.show()

