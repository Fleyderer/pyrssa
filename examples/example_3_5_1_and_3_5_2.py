import pyrssa as prs
from matplotlib import pyplot as plt
from pyrssa.conversion import collection_conversion

elec = prs.data("elec")
N = len(elec)
length = 24
L = 24
s = prs.ssa(elec[:'1993-08-01'], L=L)
si = prs.iossa(s, nested_groups=[[1, 4], [2, 3, *range(5, 11)]])
fi = prs.rforecast(si, groups={'trend': [1, 2]}, length=length, only_new=False)
s0 = prs.ssa(elec['1972-08-01':'1993-08-01'], L=L)
f0 = prs.vforecast(s0, groups={'trend': 1}, length=length, only_new=True)
si0 = prs.iossa(s0, nested_groups=[[1, 4], [2, 3, *range(5, 11)]])
fi0 = prs.vforecast(si0, groups={'trend': [1, 2]}, length=length, only_new=True)

plt.plot(elec, label='original', color='black', linewidth=0.75)
plt.plot(fi[:'1993-08-01'], label='trend', color='red', linewidth=0.75)
plt.plot(fi['1993-09-01':], label='forecast', color='blue', linewidth=1.5)
plt.plot(fi0, label='forecast0', color='lightgreen', linestyle='--', linewidth=1.5)
plt.legend()
plt.show()

L = 240
elec_sa = elec - fi
s_sa = prs.ssa(elec_sa[:'1993-08-01'], L=L)
f_sa = prs.rforecast(s_sa, groups={'trend': range(1, 14)}, length=length, only_new=False)
plt.plot(elec['1985-12-01':], label='original', color='black', linewidth=0.75)
plt.plot(fi['1985-12-01':'1993-08-01'], label='trend', color='red', linewidth=1.5)
plt.plot(fi['1993-09-01':] + f_sa['1993-09-01':], label='forecast', color='lightgreen', linewidth=1.5)
plt.show()
