import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import curvesMapping as cm
from mpl_toolkits.mplot3d import Axes3D

# ===================
# Data Preprocessing
# ===================
df = pd.read_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")
spectra = np.vstack(
    df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
)
start = df["startFrequency"].iloc[0]
end = df["endFrequency"].iloc[0]
freq = np.linspace(start, end, spectra.shape[1])
x_vals = np.sort(df["X"].unique())
y_vals = np.sort(df["Y"].unique())
peakSpectra = []
for spectrum in spectra:
    peakSpectra.append(spectrum[305:405])
peakFreq = freq[305:405]
peakSpectraarr = np.array(peakSpectra)


# ===============
# PC Computation
# ===============
PC1, PC2, PC3 = cm.compute_pcs(peakSpectra)

PC1List = []
PC2List = []
for spectrum in peakSpectra:
    PC1List.append(np.dot(spectrum, PC1))
    PC2List.append(np.dot(spectrum, PC2))
PC1arr = np.array(PC1List)
PC2arr = np.array(PC2List)


# ================================================
# VCA Endmember Extraction (3 endmembers)
# ================================================
num_endmembers = 3
endmembers, endmember_indices = cm.vca(peakSpectraarr, num_endmembers, seed=42)

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
abundances = cm.estimate_abundances(peakSpectraarr, endmembers)



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
# Y_reconstructed = abundances @ endmembers  (linear mixing model)
reconstructed = abundances @ endmembers
residuals = np.sqrt(np.mean((peakSpectraarr - reconstructed) ** 2, axis=1))

error_map = residuals.reshape(len(y_vals), len(x_vals))
plt.figure(figsize=(6, 5))
plt.imshow(error_map, origin="lower", cmap="hot")
plt.colorbar(label="RMSE")
plt.title("VCA Reconstruction Error")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.tight_layout()
plt.show()

print(f"\nReconstruction RMSE — mean: {residuals.mean():.4f}, max: {residuals.max():.4f}")
