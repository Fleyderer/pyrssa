import pyrssa as prs
from matplotlib import pyplot as plt

co2 = prs.data("co2")

s = prs.ssa(co2, L=120)

for1 = prs.rforecast(s, groups=[1, [1, 4], range(1, 5), range(1, 7)], length=12)
for series, marker, style in zip(for1, [1, 2, 3, 4], ['-', '--', ':', '-.']):
    plt.plot(series, marker=f"${marker}$", linestyle=style, linewidth=0.75)
plt.show()

for1a = prs.vforecast(s, groups={"F1": 1, "trend": [1, 4]}, length=36, only_new=False)
plt.plot(co2, color="black", linewidth=0.75)
plt.plot(for1a.trend, color="red", linewidth=0.75)
plt.show()

for2 = prs.rforecast(s, groups=[range(1, 7)], length=60, only_new=True, reverse=True)
plt.plot(co2, color="black", linewidth=0.75)
plt.plot(for2, color="red", linewidth=0.75)
plt.show()

