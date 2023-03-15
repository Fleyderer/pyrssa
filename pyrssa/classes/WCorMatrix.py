from rpy2 import robjects
import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class WCorMatrix(np.matrix):

    def __new__(cls, x, groups=None, **kwargs):

        obj = None

        if groups is None:

            if isinstance(x, np.matrix):
                groups = range(1, min(*x.shape) + 1)
                obj = x
            else:
                groups = range(1, min(x.nsigma(), x.nu()) + 1)

        if obj is None:
            obj = np.asmatrix(robjects.r.wcor(x=x, groups=groups, **kwargs)).view(cls)

        obj.groups = np.copy(groups)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.groups = getattr(obj, 'groups', None)
