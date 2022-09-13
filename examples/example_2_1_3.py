import pyrssa as prs
import pandas as pd

AustralianWine = prs.data("AustralianWine")
fort = AustralianWine['Fortified'][:174]
fort.index = pd.date_range(start='1980/01/01', freq='M', periods=len(fort))
s_fort = prs.ssa(fort, L=84, kind="1d-ssa")
r_fort = prs.reconstruct(s_fort, groups={"Trend": 1, "Seasonality": range(2, 13)})

# prs.plot(s_fort)
# prs.plot(s_fort, kind="vectors", idx=range(1, 9))
# prs.plot(s_fort, kind="paired", idx=range(2, 13), contrib=False)
# prs.plot(prs.wcor(s_fort, groups=range(1, 31)), scales=range(10, 31, 10))
prs.plot(prs.reconstruct(s_fort, groups={"G12": range(2, 4), "G4": range(4, 6),
                                         "G6": range(6, 8), "G2.4": range(8, 10)}),
         x=fort.index, method="xyplot", layout=(2, 2),
         add_original=False, add_residuals=False)
