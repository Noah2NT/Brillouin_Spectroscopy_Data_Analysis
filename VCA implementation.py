import numpy as np
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_decomposition as bd
import brillouin_plotting as bplt


# ===================
# Data Preprocessing
# ===================
spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")

#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")
x_vals, y_vals = bp.build_coordinate_arrays(df)
peakSpectraarr, peakFreq = bp.extract_frequency_window(spectra, freq, 305, 405)

bplt.plot_spectrum(peakFreq, peakSpectraarr[0])
# ===============
# PC Computation
# ===============

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(peakSpectraarr)
scores = bd.compute_pc_scores(peakSpectraarr, eigenvectors, mean_spectrum, n_components=2)
PC1arr, PC2arr = scores[:, 0], scores[:, 1]


# ================================================
# VCA Endmember Extraction (3 endmembers)
# ================================================

num_endmembers = 3
endmembers, endmember_indices = bd.vca(peakSpectraarr, num_endmembers, seed=42)

print(f"Endmember pixel indices: {endmember_indices}")
print(f"Endmember positions:")
for i, idx in enumerate(endmember_indices):
    row = df.iloc[idx]
    print(f"  Endmember {i+1}: X={row['X']}, Y={row['Y']}")


# ============================
# Plot Endmember Spectra
# ============================

plt.figure(figsize=(10, 6))
for i in range(num_endmembers):
    plt.plot(peakFreq, endmembers[i], label=f'Endmember {i+1} (pixel {endmember_indices[i]})', linewidth=1.5)
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.title("VCA Extracted Endmember Spectra")
plt.legend()
plt.tight_layout()
plt.show()


# ============================
# Abundance Estimation
# ============================

abundances = bd.estimate_abundances(peakSpectraarr, endmembers)


# ============================
# Abundance Heatmaps
# ============================

fig, axes = plt.subplots(1, num_endmembers, figsize=(5 * num_endmembers, 5))
for i in range(num_endmembers):
    abundance_map = abundances[:, i].reshape(len(y_vals), len(x_vals))
    im = axes[i].imshow(abundance_map, origin="lower", cmap="viridis")
    axes[i].set_title(f"Endmember {i+1}")
    axes[i].set_xlabel("X position")
    axes[i].set_ylabel("Y position")
    fig.colorbar(im, ax=axes[i], label="Abundance")
plt.suptitle("VCA Abundance Maps", fontsize=14)
plt.tight_layout()
plt.show()


# ================================================
# Reconstruction Error Map
# ================================================

errors = bd.compute_reconstruction_error(peakSpectraarr, endmembers, abundances)

error_map = errors.reshape(len(y_vals), len(x_vals))
plt.figure(figsize=(6, 5))
plt.imshow(error_map, origin="lower", cmap="hot")
plt.colorbar(label="L2 Error")
plt.title("VCA Reconstruction Error")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.tight_layout()
plt.show()

print(f"\nReconstruction Error — mean: {errors.mean():.4f}, max: {errors.max():.4f}")
