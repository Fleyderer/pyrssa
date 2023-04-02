import numpy as np
import pyrssa as prs

dftreering = prs.data("dftreering")

L = 300
s_tree = prs.ssa(dftreering, L=L, neig=L)
g_tree = prs.grouping_auto_pgram(s_tree, base="series", freq_bins=[0.1, 0.2, 0.3, 0.4, np.inf])
r_tree = prs.reconstruct(s_tree, groups=g_tree)
prs.plot(r_tree, method="xyplot", add_residuals=False, add_original=True)
prs.plot.spectrum(r_tree)
