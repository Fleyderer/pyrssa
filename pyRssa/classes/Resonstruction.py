from rpy2 import robjects
import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class Reconstruction:

    def __init__(self, ds, groups):
        if isinstance(groups, dict):
            for k in groups:
                groups[k] = np.asarray(groups[k]) + 1
        self.obj = r_ssa.reconstruct(ds, groups=groups)
        self.series = robjects.r.attr(self.obj, "series")
        self.residuals = robjects.r.attr(self.obj, "residuals")
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            setattr(self, name, self.obj.rx(name)[0])
