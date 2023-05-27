import os
import platform

if os.environ.get("R_HOME") is None:

    if platform.system() == "Windows":
        path = r"C:\Program Files\R"
        if os.path.exists(path):
            os.environ["R_HOME"] = os.path.join(path, os.listdir(path)[-1])
        else:
            raise FileNotFoundError("R_HOME variable does not exist")
    else:
        raise FileNotFoundError("R_HOME variable does not exist")

import rpy2.robjects.packages as rpackages
import pyrssa.installer
try:
    rpackages.importr("Rssa")
except rpackages.PackageNotInstalledError:
    pyrssa.installer.install_required()


from pyrssa.classes.SSA import SSA, IOSSA, FOSSA
from pyrssa.classes.LRR import LRR
from pyrssa.classes.Parestimate import Parestimate
from pyrssa.classes.Resonstruction import Reconstruction
from pyrssa.classes.AutoSSA import GroupPgram, GroupWCor
from pyrssa.classes.Cadzow import Cadzow
from pyrssa.classes.Forecast import RForecast, VForecast, BForecast, Forecast
from pyrssa.classes.Gapfill import Gapfill, IGapfill
from pyrssa.classes.WCorMatrix import WCorMatrix
from pyrssa.classes.HMatrix import HMatrix
from pyrssa.base import *
from pyrssa.plot import plot, clplot
