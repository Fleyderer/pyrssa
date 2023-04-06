import pyrssa as prs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

hotel = prs.data("hotel")
length = len(hotel)
n = 30
hotel_2years = hotel[:n]
s = prs.ssa(hotel_2years, L=12,
            row_projector="center",
            column_projector="center")
ios = prs.iossa(s, nested_groups=[1, range(2, 6)])
r = prs.reconstruct(ios, groups={"trend": [1]})

range_2years = np.arange(1, n + 1).reshape(-1, 1)
range_len = np.arange(1, length + 1).reshape(-1, 1)

fit_2years = LinearRegression()
fit_2years.fit(range_2years, hotel_2years)
fit_2years_continued = fit_2years.predict(range_len)
fit = LinearRegression()
fit.fit(range_len, hotel)
fit_rec = LinearRegression()
fit_rec.fit(range_2years, r.trend)
fit_rec_continued = fit_rec.predict(range_len)

plt.plot(hotel.index, hotel, "k-", linewidth=1)
plt.plot(hotel.index, fit.predict(range_len), 'g--', linewidth=2)
plt.plot(hotel.index, fit_2years_continued, 'r', linewidth=1)
plt.plot(hotel_2years.index, fit_2years.predict(range_2years), 'r', linewidth=5)
plt.plot(hotel.index, fit_rec_continued, 'b', linewidth=1)
plt.plot(hotel_2years.index, fit_rec.predict(range_2years), 'b', linewidth=5)
plt.legend(["Original series",
            "General linear trend",
            "Linear regression, forecasted",
            "Linear regression",
            "Iterative O-SSA",
            "Iterative O-SSA, forecasted"],
           loc='lower center', bbox_to_anchor=(0.5, 1.05))
plt.tight_layout()
plt.show()
