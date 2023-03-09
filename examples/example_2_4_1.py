import numpy as np
import pandas as pd
import pyrssa as prs
import matplotlib.pyplot as plt

N = 100
L = 50
omega1 = 0.07
omega2 = 0.065
omega3 = 0.15
sigma = 0.1
np.random.seed(1)
F = 2 * np.sin(2 * np.pi * omega1 * np.arange(1, N + 1)) \
    + np.sin(2 * np.pi * omega2 * np.arange(1, N + 1)) \
    + 3 * np.sin(2 * np.pi * omega3 * np.arange(1, N + 1)) \
    + sigma * np.random.normal(size=N)
s = prs.ssa(F, L)
plt.plot(F)
prs.plot(s, kind="vectors", idx=range(1, 9), layout=[2, 4])
ios = prs.iossa(s, nested_groups={"F1": range(3, 5), "F2": range(5, 7)}, maxiter=1000)
print(ios)
prs.plot(ios, kind="vectors", idx=range(1, 9), layout=[2, 4])
