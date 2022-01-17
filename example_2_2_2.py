import pyRssa
import numpy as np

n = 100
sigma = 0.5
pyRssa.seed(1)


F = np.sin(2 * np.pi * np.array(range(1, n + 1)) / 7) + sigma * np.array(pyRssa.r.rnorm(n))
F_center = F - pyRssa.r.mean(F)
st = pyRssa.ssa(F_center, L=50, kind='toeplitz-ssa')
s = pyRssa.ssa(F, L=50, kind="1d-ssa")



