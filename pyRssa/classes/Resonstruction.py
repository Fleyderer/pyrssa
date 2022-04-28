from rpy2 import robjects
import rpy2.robjects.packages as rpackages

r_ssa = rpackages.importr('Rssa')


class Reconstruct:

    def __init__(self, ds, groups):
        self.obj = r_ssa.reconstruct(ds, groups=groups)
        self.series = robjects.r.attr(self.obj, "series")
        self.trend = self.obj.rx("Trend")[0]
        self.seasonality = self.obj.rx("Seasonality")[0]
        self.residuals = robjects.r.attr(self.obj, "residuals")
