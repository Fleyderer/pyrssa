import pyrssa as prs

dwarfst = prs.data("dwarfst")

s = prs.ssa(dwarfst, L=100)
g = prs.grouping_auto(s, grouping_method="wcor",
                      method="average", nclust=2)
print(g[0])
prs.plot(prs.wcor(s, groups=range(1, 31)), scales=[1, 11, 30])
prs.plot(prs.reconstruct(s, groups=g),
         add_residuals=False,
         method="xyplot", superpose=False)
