from pyRssa import pyRssa as prs
import numpy as np

n = 100
sigma = 0.5
np.random.seed(1)

F = np.sin(2 * np.pi * np.arange(1, n + 1) / 7) + sigma * np.random.rand(n)
F_center = F - np.mean(F)
st = prs.Ssa(F_center, L=50, kind='toeplitz-ssa')
s = prs.Ssa(F, L=50, kind="1d-ssa")
