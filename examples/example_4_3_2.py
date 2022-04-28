from pyRssa import pyRssa
import numpy as np

pyRssa.seed(1)
N = 71
sigma = 5
Ls = [12, 24, 36, 48, 60]
length = 24
signal_1 = 30 * np.cos(2 * pyRssa.pi * pyRssa.seq(1, N + length) / 12)
signal_2 = 30 * np.cos(2 * pyRssa.pi * pyRssa.seq(1, N + length) / 12 + pyRssa.pi / 4)
signal = pyRssa.r.cbind(signal_1, signal_2)
R = 100


def errors(Ls):
    f1 = signal_1[:N] + pyRssa.r.rnorm(N, sd=sigma)
    f2 = signal_2[:N] + pyRssa.r.rnorm(N, sd=sigma)
    f = pyRssa.r.cbind(f1, f2)
    err_rec = pyRssa.r.numeric(5)
    err_for = pyRssa.r.numeric(5)
    for l in range(len(Ls)):
        L = Ls[l]
        s = pyRssa.ssa(f, L=L, kind="mssa")
        rec = pyRssa.reconstruct(s, groups=pyRssa.list([1, 2]))[0]
        err_rec[l] = pyRssa.mean((rec - signal[:N]) ** 2)
        pred = pyRssa.vforecast(s, groups=pyRssa.list([1, 2]), direction="row", len=length, drop=True)
        err_for[l] = pyRssa.mean((pred - signal[N:]) ** 2)
    return pyRssa.list(Reconstruction=err_rec, Forecast=err_for)


res = pyRssa.replicate(R, errors, Ls)
err_rec = pyRssa.r.rowMeans(pyRssa.r.simplify2array(pyRssa.rows(res, "Reconstruction")))
err_for = pyRssa.r.rowMeans(pyRssa.r.simplify2array(pyRssa.rows(res, "Forecast")))




