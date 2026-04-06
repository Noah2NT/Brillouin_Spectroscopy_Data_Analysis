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

folded_spectra, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)

keep = shift_freq <= 8
folded_spectra = folded_spectra[:, keep]
shift_freq = shift_freq[keep]
# ===================
# Fitting
# ===================

fitted_curves, fit_params = bf.fit_all_spectra(shift_freq, folded_spectra)
amplitudes, peak_shifts, linewidths, baselines = bf.extract_fit_parameters(fit_params)

shift_map = bp.reshape_to_map(peak_shifts, y_vals, x_vals)

# ===================
# VCA Endmember Abundance
# ===================

endmembers, indices = bd.vca(folded_spectra, num_endmembers=3)
abundances = bd.estimate_abundances(folded_spectra, endmembers)
endmember2_map = bp.reshape_to_map(abundances[:, 0], y_vals, x_vals)

# ===================
# Morphological Mask Comparison
# ===================

mask = fe.threshold_mask(endmember2_map, np.mean(endmember2_map))
eroded = fe.erode_mask(mask, iterations=1)
dilated = fe.dilate_mask(eroded, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mask, cmap='gray', interpolation='nearest', origin='lower')
axes[0].set_title('Original Mask')
axes[0].axis('off')

axes[1].imshow(eroded, cmap='gray', interpolation='nearest', origin='lower')
axes[1].set_title('After Erosion')
axes[1].axis('off')

axes[2].imshow(dilated, cmap='gray', interpolation='nearest', origin='lower')
axes[2].set_title('After Dilation')
axes[2].axis('off')

plt.suptitle('Morphological Operations on VCA Endmember 3 Abundance Mask', fontsize=14)
plt.tight_layout()
plt.show()
