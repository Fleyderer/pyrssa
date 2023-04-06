# import rpy2's package module
import rpy2.robjects.packages as rpackages
# R vector of strings
from rpy2.robjects.vectors import StrVector
# import R's utility package
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list


def install_required():
    pack_names = ('Rssa', 'stats', )
    names_to_install = [x for x in pack_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(
            StrVector(names_to_install))
