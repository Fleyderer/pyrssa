import pyRssa

AustralianWine = pyRssa.data("AustralianWine", package="Rssa")
wine = pyRssa.window(AustralianWine, end=pyRssa.time(AustralianWine)[173])
fort = pyRssa.cols(wine, "Fortified")
s_fort = pyRssa.ssa(fort, L=84, kind="1d-ssa")
r_fort = pyRssa.reconstruct(s_fort, groups=pyRssa.list(Trend=1, Seasonality=range(2, 12)))

pyRssa.parestimate(s_fort, group=pyRssa.list(range(2, 4), range(4, 6)), method="pairs")
pyRssa.wcor(s_fort, groups=range(1, 31))
rec = pyRssa.reconstruct(s_fort, groups=pyRssa.list(G12=range(2, 4), G4=range(4, 6), G6=range(6, 8), G2_4=range(8, 10)))

import matplotlib.pyplot as plt

pyRssa.mplot(r_fort, X=pyRssa.time(AustralianWine)[:174], add_original=True, add_residuals=True)

