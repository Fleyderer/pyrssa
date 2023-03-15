import numpy as np
from rdatasets import data
import pyrssa as prs

co2 = data("co2")
s2 = prs.ssa(co2.value, column_projector="centering", row_projector="centering")
prs.plot(prs.reconstruct(s2, groups={"Linear_trend": range(1, s2.nspecial() + 1)}),
         add_residuals=False, method="matplot")
s4 = prs.ssa(co2.value, column_projector=2, row_projector=2)
prs.plot(prs.reconstruct(s4, groups={"Linear_trend": range(1, s4.nspecial() + 1)}),
         add_residuals=False, method="matplot")
prs.plot(s4, kind="vectors", idx=range(1, 13))
r = prs.reconstruct(s4, groups={"Signal": np.concatenate((range(1, s4.nspecial() + 1), range(5, 9)))})
prs.plot(r, method="xyplot")
