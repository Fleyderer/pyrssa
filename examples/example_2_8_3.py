import pyrssa as prs

MotorVehicle = prs.data("MotorVehicle")
s1 = prs.ssa(MotorVehicle, L=12)
res1 = prs.reconstruct(s1, groups={"trend": 1})
trend = res1.trend
prs.plot(res1, add_residuals=False, color=["black", "red"], linewidth=[0.75, 1.5], superpose=True, method="xyplot")
res_trend = res1.residuals
s2 = prs.ssa(res_trend, L=264)
res2 = prs.reconstruct(s2, groups={"seasonality": range(1, 11)})
seasonality = res2.seasonality
