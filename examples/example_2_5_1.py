import numpy as np
import pyrssa as prs

N = 100
L = 50
omega1 = 0.03
omega2 = 0.06
sigma = 0.1
np.random.seed(3)

n = np.arange(1, N + 1)
F = np.sin(2 * np.pi * omega1 * n) + \
    np.sin(2 * np.pi * omega2 * n) + \
    sigma * np.random.normal(size=N)
s = prs.ssa(F, L=L, neig=min(L, N - L + 1))
prs.plot(s)

fos = prs.fossa(s, nested_groups=[[1, 2], [3, 4]], gamma=10, normalize=False)
ios1 = prs.iossa(s, nested_groups=[[1, 2], [3, 4]], maxiter=1)
ios2 = prs.iossa(s, nested_groups=[[1, 2], [3, 4]], maxiter=2)

prs.plot(s, kind="vectors", idx=range(1, 5), layout=(1, 4),
         title="Eigenvectors, Basic SSA")
prs.plot(fos, kind="vectors", idx=range(1, 5), layout=(1, 4),
         title="Eigenvectors, SSA with derivatives")
prs.plot(ios1, kind="vectors", idx=range(1, 5), layout=(1, 4),
         title="Eigenvectors, Iterative O-SSA, 1 iter")
prs.plot(ios2, kind="vectors", idx=range(1, 5), layout=(1, 4),
         title="Eigenvectors, Iterative O-SSA, 2 iter")

print(np.sum(fos.sigma ** 2) / np.sum(s.sigma ** 2) * 100)
print(np.sum(ios1.sigma ** 2) / np.sum(s.sigma ** 2) * 100)
print(np.sum(ios2.sigma ** 2) / np.sum(s.sigma ** 2) * 100)

fo_rec = prs.reconstruct(fos, groups=[[1, 2], [3, 4]])
io_rec = prs.reconstruct(ios2, groups=ios2.iossa_groups)

prs.plot(fo_rec, method="xyplot", title="SSA with derivatives")
prs.plot(io_rec, method="xyplot", title="Iterative O-SSA")
