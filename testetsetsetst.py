import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt
import feature_extraction as fe

# ===================
# Data Loading
# ===================

spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv") # needs freqShift > 8 excluded
#spectra, freq, df = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")
#spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")
#spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")

x_vals, y_vals = bp.build_coordinate_arrays(df)

# ===================
# Raw Spectrum with Stokes Peak Marker
# ===================

spectrum = spectra[0]

stokes_peak_idx = np.argmax(spectrum[300:400])
print(stokes_peak_idx)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freq[300:400], spectrum[300:400], linewidth=1.2)
ax.plot(freq[300 + stokes_peak_idx], spectrum[300+stokes_peak_idx],
        'rv', markersize=14, zorder=5)
ax.set_xlabel('Frequency Shift (GHz)')
ax.set_ylabel('Intensity')
plt.tight_layout()
plt.show()

# ===================
# Stokes / Anti-Stokes Folding
# ===================
