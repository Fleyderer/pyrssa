import pyrssa as prs
import pandas as pd

AustralianWine = prs.data("AustralianWine")
fort = AustralianWine['Fortified'][:174]
print(fort)
print(type(fort))
fort.index = pd.date_range(start='1980/01/01', freq='M', periods=len(fort))
s_fort = prs.ssa(fort, L=84, kind="1d-ssa")
r_fort = prs.reconstruct(s_fort, groups={"Trend": 0, "Seasonality": range(1, 11)})
prs.plot(r_fort, x=fort.index, add_original=True, add_residuals=True, method="xyplot", superpose=True)
