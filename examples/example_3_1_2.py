import pyrssa as prs

# Decompose "co2" series with default window length L
co2 = prs.data("co2")
# Estimate the periods from 2nd and 3rd eigenvectors
# using default "pairs" method
s = prs.ssa(co2)
print(prs.parestimate(s, groups=[[2, 3]], method="pairs"))
# Estimate the periods and rates using ESPRIT
pe = prs.parestimate(s, groups=[range(1, 7)], method="esprit")
print(pe)
prs.plot(pe)
