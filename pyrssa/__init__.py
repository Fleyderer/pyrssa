import os
import platform
# TODO: change behavior of searching for R (maybe by independently installing R in pyrssa).
if os.environ.get("R_HOME") is None:
    if platform.system() == "Windows":
        os.environ["R_HOME"] = os.path.join(os.path.dirname(__file__), "..", r"r\win")


from pyrssa.classes.SSA import SSA, IOSSA, FOSSA
from pyrssa.classes.LRR import LRR
from pyrssa.classes.Parestimate import Parestimate
from pyrssa.classes.Resonstruction import Reconstruction
from pyrssa.classes.AutoSSA import GroupPgram, GroupWCor
from pyrssa.classes.Forecast import RForecast, VForecast, BForecast
from pyrssa.classes.WCorMatrix import WCorMatrix
from pyrssa.classes.HMatrix import HMatrix
from pyrssa.base import *
from pyrssa.plot import plot, clplot
