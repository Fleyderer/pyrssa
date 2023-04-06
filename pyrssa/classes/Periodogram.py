import rpy2.robjects.packages as rpackages
import numpy as np
from functools import cached_property

r_stats = rpackages.importr('stats')


class Periodogram:

    def __init__(self, x, spans=None, kernel=None, taper=0.1,
                 pad=0, fast=True, demean=False, detrend=True, **kwargs):

        self.obj = r_stats.spec_pgram(x, spans=spans, kernel=kernel, taper=taper, pad=pad,
                                      fast=fast, demean=demean, detrend=detrend, plot=False, **kwargs)
        self.series = x
        self.detrend = detrend
        self.demean = demean
        self.taper = taper
        self.pad = pad

    @cached_property
    def freq(self):
        return np.asarray(self.obj.rx("freq")[0])

    @cached_property
    def spec(self):
        return np.asarray(self.obj.rx("spec")[0])

    @cached_property
    def coh(self):
        return np.asarray(self.obj.rx("coh")[0])

    @cached_property
    def phase(self):
        return np.asarray(self.obj.rx("phase")[0])

    @cached_property
    def kernel(self):
        return np.asarray(self.obj.rx("kernel")[0])

    @cached_property
    def df(self):
        return self.obj.rx("df")[0][0]

    @cached_property
    def bandwidth(self):
        return self.obj.rx("bandwidth")[0][0]

    @cached_property
    def n_used(self):
        return self.obj.rx("n.used")[0][0]

    @cached_property
    def orig_n(self):
        return self.obj.rx("orig.n")[0][0]

    @cached_property
    def snames(self):
        return np.asarray(self.obj.rx("snames")[0])

    @cached_property
    def method(self):
        return self.obj.rx("method")[0][0]

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
