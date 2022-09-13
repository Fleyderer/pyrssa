from pyrssa import base
import numpy as np

base.seed(1)
N = 71
sigma = 5
Ls = [12, 24, 36, 48, 60]
length = 24
signal_1 = 30 * np.cos(2 * np.pi * base.seq(1, N + length) / 12)
signal_2 = 30 * np.cos(2 * np.pi * base.seq(1, N + length) / 12 + np.pi / 4)
signal = base.r.cbind(signal_1, signal_2)
R = 100


def errors(Ls):
    f1 = signal_1[:N] + base.r.rnorm(N, sd=sigma)
    f2 = signal_2[:N] + base.r.rnorm(N, sd=sigma)
    f = base.r.cbind(f1, f2)
    err_rec = base.r.numeric(5)
    err_for = base.r.numeric(5)
    for l in range(len(Ls)):
        L = Ls[l]
        s = base.ssa(f, L=L, kind="mssa")
        rec = base.reconstruct(s, groups=base.list([1, 2]))[0]
        err_rec[l] = base.mean((rec - signal[:N]) ** 2)
        pred = base.vforecast(s, groups=base.list([1, 2]), direction="row", len=length, drop=True)
        err_for[l] = base.mean((pred - signal[N:]) ** 2)
    return {"Reconstruction":err_rec, "Forecast":err_for}


res = base.replicate(R, errors, Ls)
err_rec = base.r.rowMeans(base.r.simplify2array(base.rows(res, "Reconstruction")))
err_for = base.r.rowMeans(base.r.simplify2array(base.rows(res, "Forecast")))




