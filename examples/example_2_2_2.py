from pyrssa import base
import numpy as np

n = 100
sigma = 0.5
base.seed(1)


F = np.sin(2 * np.pi * np.array(range(1, n + 1)) / 7) + sigma * np.array(base.r.rnorm(n))
F_center = F - base.r.mean(F)
st = base.ssa(F_center, L=50, kind='toeplitz-ssa')
s = base.ssa(F, L=50, kind="1d-ssa")



