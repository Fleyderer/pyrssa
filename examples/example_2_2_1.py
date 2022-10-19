import pyrssa as prs
import numpy as np

n = 100
sigma = 0.5
np.random.seed(42)

F = np.sin(2 * np.pi * np.arange(1, n + 1) / 7) + sigma * np.random.rand(n)
F_center = F - np.mean(F)
st = prs.ssa(F_center, L=50, kind='toeplitz-ssa')
s = prs.ssa(F, L=50, kind="1d-ssa")
prs.plot(s, kind="vectors", idx=range(1, 9), layout=[1, 4])
print(s.U)
prs.plot(st, kind="vectors", idx=range(1, 5), layout=[1, 4])
prs.plot(prs.reconstruct(st, groups=[range(1, 3)]), layout=[1, 3])
prs.plot(prs.reconstruct(s, groups=[range(1, 3)]), layout=[1, 3])
