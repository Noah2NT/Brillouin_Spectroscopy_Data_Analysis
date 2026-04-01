import numpy as np
from sklearn.cluster import KMeans

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt


'''
run PCA
find n num of components that explains at least 95% of the variance to estimate how many components exist
run VCA with n endmembers
extract meaningful features from VCA images
'''


# ===================
# Data Preprocessing
# ===================

#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")     # clustering doesnt work
#spectra, freq, df  = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")              # needs fitting before separation algorithms
#spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")        # 3 clusters, azim 100

spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")
x_vals, y_vals             = bp.build_coordinate_arrays(df)
peakSpectraarr, peakFreq   = bp.extract_frequency_window(spectra, freq, int(len(freq)/20*11), len(freq))

# ========================
# Find Relevant Components
# ========================

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(peakSpectraarr)
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance      = np.cumsum(explained_variance_ratio)
n_endmembers             = int(np.searchsorted(cumulative_variance, 0.95) + 1)
n_endmembers = 2 if n_endmembers == 1 else n_endmembers
print(f"Components to explain 95% variance: {n_endmembers}")
print(f"Cumulative variance at {n_endmembers} components: {cumulative_variance[n_endmembers - 1]:.4f}")

# ========================
# VCA Endmember Extraction
# ========================

endmembers, endmember_indices = bd.vca(peakSpectraarr, n_endmembers, seed=69)
abundances = bd.estimate_abundances(peakSpectraarr, endmembers)
endmember_pos = []
for i, idx in enumerate(endmember_indices):
    row = df.iloc[idx]
    endmember_pos.append((row['X'], row['Y']))

# ===================================
# 3D PCA Scatter with VCA Endmembers
# ===================================

scores = bd.compute_pc_scores(peakSpectraarr, eigenvectors, mean_spectrum, n_components=3)
PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]

bplt.plot_3d_colored_scatter(
    PC1arr, PC2arr, PC3arr,
    colors=abundances[:, 0],
    xlabel="PC1", ylabel="PC2", zlabel="PC3",
    title="PCA Space — VCA Endmembers Highlighted",
    cmap="inferno",
    endmember_indices=endmember_indices,
)


# ===================================
# VCA Abundance Maps with Endmembers
# ===================================

bplt.plot_component_maps(
    arrays=[abundances[:, i] for i in range(n_endmembers)],
    x_vals=x_vals, y_vals=y_vals,
    labels=[f"Endmember {i+1}" for i in range(n_endmembers)],
    suptitle="VCA Abundance Maps",
    cmap="inferno", colorbar_label="Abundance", centres=endmember_pos,
)



# ==================
# Abundance 2D Maps
# ==================

abundance_maps = [
    abundances[:, i].reshape(len(y_vals), len(x_vals))
    for i in range(n_endmembers)
]
print(abundance_maps[0])
#asa

# ==================
# Feature Extraction
# ==================

