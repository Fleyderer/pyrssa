import numpy as np
import pyrssa as prs
from matplotlib import pyplot as plt

USUnemployment = prs.data("USUnemployment")
ser = USUnemployment["MALE"]
Time = ser.index
L = 204

ss = prs.ssa(ser, L=L, svd_method="eigen")
res = prs.reconstruct(ss, groups=[list(range(1, 5)) + list(range(7, 12)), [5, 6, 12, 13]])
trend, seasonality = res.F1, res.F2
w1 = prs.wcor(ss, groups=range(1, 31))

fss = prs.fossa(ss, nested_groups=[list(range(1, 5)) + list(range(7, 12)), [5, 6, 12, 13]],
                gamma=np.inf)
fres = prs.reconstruct(fss, groups=[range(5, 14), range(1, 5)])
ftrend, fseasonality = fres.F1, fres.F2

plt.plot(Time, ser, color="grey", linewidth=2)
plt.plot(Time, trend, color="blue", linewidth=1)
plt.plot(Time, ftrend, color="red", linewidth=1)
plt.legend(["Full series", "Basic SSA trend", "DerivSSA trend"])
plt.show()

plt.plot(Time, seasonality, color="blue", linewidth=2)
plt.plot(Time, fseasonality, color="red", linewidth=1)
plt.legend(["Basic SSA seasonality", "DerivSSA seasonality"])
plt.show()
