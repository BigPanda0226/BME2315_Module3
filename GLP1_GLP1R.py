
# GLP1_GLP1R.py

import numpy as np
import matplotlib.pyplot as plt

from ligand_receptor import Fact

# GLP-1 concentrations [nM]
GLP1_FASTING = 0.0150
GLP1_FED = 0.040

# Semaglutide concentrations [nM]
semaglutide = np.logspace(-2, 4, 500)

# Calculate receptor activation
response_fasting = Fact(GLP1_FASTING, semaglutide)
response_fed = Fact(GLP1_FED, semaglutide)

# Plot concentration-response curves
plt.figure()

plt.semilogx(semaglutide, response_fasting, 
             label="Fasting GLP-1", lw=2)

plt.semilogx(semaglutide, response_fed, 
             label="Post-meal GLP-1", lw=2)

plt.ylim(0, 1)
plt.xlabel("Semaglutide concentration (nM)")
plt.ylabel("Fraction of GLP1R activated")
plt.legend()
plt.savefig("GLP1_GLP1R.png", dpi=300, bbox_inches="tight")
plt.show()
