import numpy as np
import pyrssa as prs

# minimal LRR
x = 1.01 ** np.arange(1, 11)
s = prs.ssa(x, L=2)
l = prs.lrr(s, groups=[1])
print(l)
print(l.roots())

# extraneous roots
x = 1.01 ** np.arange(1, 11)
s = prs.ssa(x, L=6)
l = prs.lrr(s, groups=[1])
r = l.roots()
prs.plot(l)

# multiple roots
x = 2.188 * np.arange(1, 11) + 7.77
s = prs.ssa(x, L=3)
l = prs.lrr(s, groups=[[1, 2]])
print(l)
print(l.roots())
