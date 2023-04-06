import pyrssa as prs
from matplotlib import pyplot as plt
import numpy as np

paynsa = prs.data("paynsa")
n = 241
pay = paynsa[n - 1:]
s = prs.ssa(pay, L=36)
g1 = prs.grouping_auto(s, base="series",
                       freq_bins={"trend": 0.06}, threshold=0.7)
print(g1.trend)
prs.plot(g1, order=True, marker='o')
r1 = prs.reconstruct(s, g1)
prs.plot(r1, method="xyplot", superpose=True, add_residuals=False)
s1 = prs.ssa(pay - r1.trend, L=120)
coef = np.array([1 - 0.02, 1 + 0.02])
freq_bins_seas = {"s12": 1/12 * coef, "s6": 1/6 * coef,
                  "s4": 1/4 * coef, "s3": 1/3 * coef,
                  "s2.4": 1/2.4 * coef, "s2": 1/2 * coef}
g3 = prs.grouping_auto(s1, base="series", groups=range(1, 21),
                       freq_bins=freq_bins_seas, threshold=[0.6])
prs.plot(g3, order=True, scales=None, legend_params={"ncol": 3},
         marker='o', markerfacecolor='None', linestyle='None')
prs.plot(g3, order=False, scales=None, legend_params={"ncol": 3},
         marker='o', markerfacecolor='None', linestyle='None')
r3 = prs.reconstruct(s1, groups=[[g_item for group in g3 for g_item in group]])
prs.plot(r3, method="xyplot", add_residuals=False, add_original=False)
specNSA = prs.spectrum(pay - r3.F1)
specSA = prs.spectrum(pay)
plt.plot(specNSA.freq, np.log(specNSA.spec))
plt.plot(specSA.freq, np.log(specSA.spec))
plt.show()
