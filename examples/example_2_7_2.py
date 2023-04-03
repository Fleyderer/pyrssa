import numpy as np
import pyrssa as prs

oilproduction = prs.data("oilproduction")
s = prs.ssa(oilproduction, L=120)
g0 = prs.grouping_auto(s, base="series", freq_bins={"Tendency": 1/240, "Trend": 1/24}, threshold=0.1)
prs.plot(g0, order=True, marker='o')
contrib = g0.contributions.iloc[:, 1]
print(thr := contrib.sort_values(ascending=False).iloc[8])
g = prs.grouping_auto(s, base="series", freq_bins={"Tendency": 1/240, "Trend": 1/24}, threshold=thr)
print(g[0])
print(g[1])
prs.plot(prs.reconstruct(s, groups=g), add_residuals=False, method="xyplot", superpose=True)
