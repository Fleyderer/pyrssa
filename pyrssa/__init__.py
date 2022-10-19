import os
if os.environ.get("R_HOME") is None:
    # Change this path as needed
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"


from pyrssa.classes.SSA import SSA
from pyrssa.classes.Parestimate import Parestimate
from pyrssa.classes.Resonstruction import Reconstruction
from pyrssa.classes.Forecast import *
from pyrssa.classes.WCorMatrix import WCorMatrix
from pyrssa.base import *
