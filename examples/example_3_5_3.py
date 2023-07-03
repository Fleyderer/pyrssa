import pyrssa as prs
import numpy as np
from matplotlib import pyplot as plt

cowtemp = prs.data('cowtemp')
series = cowtemp
N = len(series)
cut = 14
future = 21
length = cut + future
r = 1
L = 28
Lt = 14
s = prs.ssa(series[:N - cut], L=L)
print(prs.parestimate(s, groups={'trend': range(1, r + 1)}, method='esprit').moduli)
print(prs.lrr(s, groups={'trend': range(1, r + 1)}).roots()[0])
rec = prs.reconstruct(s, groups=range(1, r + 1))
st = prs.ssa(series[:N - cut], kind='toeplitz-ssa', L=Lt)
print(prs.parestimate(st, groups={'trend': range(1, r + 1)}, method='esprit').moduli)
print(prs.parestimate(st, groups={'trend': range(1, r + 1)}, normalize_roots=False, method='esprit').moduli)
print(prs.lrr(st, groups={'trend': range(1, r + 1)}).roots()[0])
fr = prs.rforecast(s, groups={'trend': range(1, r + 1)}, length=length, only_new=True)
fv = prs.vforecast(s, groups={'trend': range(1, r + 1)}, length=length, only_new=False)
ftr = prs.rforecast(st, groups={'trend': range(1, r + 1)}, length=length, only_new=False)
print(np.sqrt(np.mean((fr.loc[61:] - series[61:]) ** 2)))
print(np.sqrt(np.mean((fv.loc[61:] - series[61:]) ** 2)))
print(np.sqrt(np.mean((ftr.loc[61:] - series[61:]) ** 2)))

plt.plot(series, label='original series', color='black', linewidth=0.75)
plt.plot(rec.F1, label='reconstructed series', color='red', linewidth=0.75)
plt.plot(fr, label='recurrent forecast', color='blue', linewidth=1.5)
plt.plot(fv, label='vector forecast', color='lightgreen', linestyle='--', linewidth=1.5)
plt.plot(ftr, label='recurrent Toeplitz forecast', color='violet', linewidth=0.75)
plt.legend()
plt.show()