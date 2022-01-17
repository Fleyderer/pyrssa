import pyRssa

AustralianWine = pyRssa.data("AustralianWine", package="Rssa")
wine = pyRssa.window(AustralianWine, end=pyRssa.time(AustralianWine)[173])
fort = pyRssa.cols(wine, "Fortified")
s_fort = pyRssa.ssa(fort, L=84, kind="1d-ssa")
r_fort = pyRssa.reconstruct(s_fort, groups=pyRssa.list(Trend=1, Seasonality=range(2, 12)))


