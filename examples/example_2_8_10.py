import pyrssa as prs
from matplotlib import pyplot as plt
import numpy as np

elec = prs.data("elec")
elec_log = np.log(elec)
Time = elec.index
s = prs.ssa(elec, L=12)
r = prs.reconstruct(s, groups={"trend": [1]})
sl = prs.ssa(elec_log, L=12)
rl = prs.reconstruct(sl, groups={"trend": [1]})
plt.plot(Time, elec, label="original")
plt.plot(Time, np.exp(rl.trend), label="exp(log-trend)")
plt.plot(Time, r.trend, label="trend")
plt.plot(Time, elec - r.trend, label="residual")
plt.legend()
plt.show()
