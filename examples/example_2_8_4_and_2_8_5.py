import numpy as np
import pyrssa as prs
from matplotlib import pyplot as plt

MotorVehicle = prs.data("MotorVehicle")
s = prs.ssa(MotorVehicle, L=264)
sf = prs.fossa(s, nested_groups=range(1, 20))
rf = prs.reconstruct(sf, groups={"seasonality": range(1, 11),
                                 "trend": range(11, 20)})
prs.plot(rf, method="xyplot", superpose=True, add_residuals=False,
         color=["black", "darkgreen", "red"], linewidth=[0.75, 0.75, 1.5])
p = prs.parestimate(sf, groups=[range(1, 11)], method="esprit")

print(*[f"{x:.2f}" for x in p.periods[0:10:2]])

resf = rf.residuals
s_env = prs.ssa(resf ** 2, L=30)
rsd = np.sqrt(prs.reconstruct(s_env, groups=[1]).F1)
plt.plot(resf)
plt.plot(rsd)
plt.plot(-rsd)
plt.legend(["resf", "rsd", "-rsd"])
plt.show()
