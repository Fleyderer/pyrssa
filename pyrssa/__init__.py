import os
# TODO: change behavior of searching for R (maybe by independently installing R in pyrssa).
if os.environ.get("R_HOME") is None:
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"


from pyrssa.classes.SSA import SSA, IOSSA
from pyrssa.classes.Parestimate import Parestimate
from pyrssa.classes.Resonstruction import Reconstruction
from pyrssa.classes.Forecast import *
from pyrssa.classes.WCorMatrix import WCorMatrix
from pyrssa.base import *
