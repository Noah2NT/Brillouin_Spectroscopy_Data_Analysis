import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import RBFInterpolator

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd


# ===================
# Data Preprocessing
# ===================
#spectra, freq, df = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")
#spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")
#spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")
spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")
x_vals, y_vals = bp.build_coordinate_arrays(df)
peakSpectraarr, peakFreq = bp.extract_frequency_window(spectra, freq, 305, 405)


# ===============
# PC Computation
# ===============

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(peakSpectraarr)
scores = bd.compute_pc_scores(peakSpectraarr, eigenvectors, mean_spectrum, n_components=3)
PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]


# ============================
# Principal Component Vectors
# ============================

n_show = min(10, len(eigenvalues))
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(2):
    ax.plot(peakFreq, eigenvectors[:, i], linewidth=1.5, label=f'PC{i+1} ({eigenvalues[i]/eigenvalues.sum()*100:.1f}%)')
ax.set_xlabel('Frequency Shift (GHz)')
ax.set_ylabel('Eigenvector Weight')
ax.set_title('Principal Components')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ============================
# PCA Abundance Heatmaps
# ============================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
PC_labels = ['PC1', 'PC2']
PC_arrays = [PC1arr, PC2arr]

for i in range(2):
    heatmap = PC_arrays[i].reshape(len(y_vals), len(x_vals))
    im = axes[i].imshow(heatmap, origin="lower", cmap='magma')
    axes[i].set_title(f"{PC_labels[i]} Score Map")
    axes[i].set_xlabel("X position")
    axes[i].set_ylabel("Y position")
    fig.colorbar(im, ax=axes[i], label=f"{PC_labels[i]} score")
plt.suptitle("PCA Abundance Maps", fontsize=14)
plt.tight_layout()
plt.show()


# ===================
# Compute Peak Shift
# ===================

fitted_curves, fit_params = bf.fit_all_spectra(peakFreq, peakSpectraarr)
_, freqShiftarr, _, _ = bf.extract_fit_parameters(fit_params)


# ===================
# Plot according to peak shift in PC space (good) (noticable gradient)
# ===================
'''
i = 0  # plots in 3d from all angles
while i < 360:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(PC1arr, PC2arr, PC3arr, c=freqShiftarr, cmap='magma', s=10)
    plt.colorbar(sc, label='Frequency shift')
    ax.set_title(str(i))
    ax.set_xlabel('PC1 value')
    ax.set_ylabel('PC2 value')
    ax.set_zlabel('PC3 value')
    ax.view_init(elev=0, azim=i)
    plt.show()
    i += 10
'''

fig = plt.figure()  # plots in 3d at the clearest angle
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(PC1arr, PC2arr, PC3arr, c=freqShiftarr, cmap='magma', s=10)
plt.colorbar(sc, label='Frequency Shift (GHz)')
ax.set_title("Brillouin data in PC space")
ax.set_xlabel('PC1 value')
ax.set_ylabel('PC2 value')
ax.set_zlabel('PC3 value')
ax.view_init(elev=0, azim=210)
plt.show()


plt.figure()  # plots in 2d
sc = plt.scatter(PC1arr, PC2arr, c=freqShiftarr, cmap='magma', s=20)
plt.colorbar(label='Frequency Shift (GHz)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Scatter Plot')
plt.show()


# ===================
# Plot contour lines for frequency shift on PC space chart
# ===================

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(PC1arr, PC2arr, c=freqShiftarr, cmap='magma', s=20, alpha=0.6)
plt.colorbar(sc, label='Frequency Shift (GHz)')
pc1_grid = np.linspace(PC1arr.min(), PC1arr.max(), 200)
pc2_grid = np.linspace(PC2arr.min(), PC2arr.max(), 200)
PC1g, PC2g = np.meshgrid(pc1_grid, pc2_grid)
points = np.column_stack([PC1arr, PC2arr])
grid_points = np.column_stack([PC1g.ravel(), PC2g.ravel()])
rbf = RBFInterpolator(points, freqShiftarr, smoothing=1.0)
shift_grid = rbf(grid_points).reshape(PC1g.shape)
contour = ax.contour(PC1g, PC2g, shift_grid, levels=10, colors='black', linewidths=3, zorder=5)
ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f GHz')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Frequency Shift Gradient in PC Space')
plt.tight_layout()
plt.show()
