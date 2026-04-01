import numpy as np
from sklearn.cluster import KMeans

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt


# ===================
# Data Preprocessing
# ===================

spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")                                      # 2 clusters, azim 110
#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")     # clustering doesnt work
#spectra, freq, df  = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")              # needs fitting before separation algorithms
#spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")        # 3 clusters, azim 100

x_vals, y_vals             = bp.build_coordinate_arrays(df)
peakSpectraarr, peakFreq   = bp.extract_frequency_window(spectra, freq, 305, 512)

# ===============
# PCA
# ===============

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(peakSpectraarr)
scores = bd.compute_pc_scores(peakSpectraarr, eigenvectors, mean_spectrum, n_components=3)  # (N, 3)
PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]


# =========================
# VCA Endmember Extraction
# =========================

num_endmembers = 2
endmembers, endmember_indices = bd.vca(peakSpectraarr, num_endmembers, seed=42)
abundances = bd.estimate_abundances(peakSpectraarr, endmembers)

print(f"Endmember pixel indices: {endmember_indices}")
print("Endmember positions:")
for i, idx in enumerate(endmember_indices):
    row = df.iloc[idx]
    print(f"  Endmember {i+1}: X={row['X']}, Y={row['Y']}")


# ============================
# PCA Score Maps
# ============================

bplt.plot_component_maps(
    arrays=[PC1arr, PC2arr, PC3arr],
    x_vals=x_vals, y_vals=y_vals,
    labels=["PC1 Score Map", "PC2 Score Map", "PC3 Score Map"],
    suptitle="PCA Score Maps",
    cmap="magma", colorbar_label="PC score",
)


# ============================
# VCA Abundance Maps
# ============================

bplt.plot_component_maps(
    arrays=[abundances[:, i] for i in range(num_endmembers)],
    x_vals=x_vals, y_vals=y_vals,
    labels=[f"Endmember {i+1}" for i in range(num_endmembers)],
    suptitle="VCA Abundance Maps",
    cmap="inferno", colorbar_label="Abundance",
)


# ===================
# k-means Clustering
# ===================

X = np.column_stack((PC1arr, PC2arr, PC3arr))
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

bplt.plot_heatmap(
    labels, x_vals, y_vals,
    title="Cluster Heatmap", cmap="jet", colorbar_label="Cluster",
)


# ====================
# Scatter in PCA space (3D)
# ====================

bplt.plot_3d_colored_scatter(
    PC1arr, PC2arr, PC3arr,
    colors=labels,
    xlabel="PC1", ylabel="PC2", zlabel="PC3",
    title="PCA Scatter Plot",
    elev=40, azim=270,
)


# ============================
# Endmember Spectra
# ============================

bplt.plot_multiple_spectra(
    peakFreq, endmembers,
    labels=[f"Endmember {i+1} (pixel {endmember_indices[i]})"
            for i in range(num_endmembers)],
    title="VCA Extracted Endmember Spectra",
)
