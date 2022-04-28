from pyRssa import pyRssa as prs
import pandas as pd

AustralianWine = prs.data("AustralianWine")
fort = AustralianWine['Fortified'][:174]
fort.index = pd.date_range(start='1980/01/01', freq='M', periods=len(fort))
s_fort = prs.Ssa(fort, L=84, kind="1d-ssa")
r_fort = prs.Reconstruct(s_fort, groups={"Trend": 1, "Seasonality": range(2, 12)})

prs.plot(s_fort)
prs.plot(s_fort, kind="vectors", idx=range(8))
prs.plot(s_fort, kind="paired", idx=range(1, 11))

