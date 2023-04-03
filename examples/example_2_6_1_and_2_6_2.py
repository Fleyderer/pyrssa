import numpy as np
import pyrssa as prs

F = prs.data("co2")
F.iloc[99:200] = np.nan
prs.clplot(F)

s1 = prs.ssa(F, L=72)
prs.plot(s1, kind="vectors", idx=range(1, 13))
prs.plot(s1, kind="series", groups=range(1, 7), layout=(3, 2))
prs.plot(prs.wcor(s1, groups=range(1, 21)), scales=range(1, 21, 2))
prs.plot(prs.reconstruct(s1, groups=[[1, 4, 7]]), add_residuals=False, method="xyplot", superpose=True)

s2 = prs.ssa(F, L=120)
prs.plot(prs.reconstruct(s2, groups=[[1, 6, 7]]), add_residuals=False, method="xyplot", superpose=True)
