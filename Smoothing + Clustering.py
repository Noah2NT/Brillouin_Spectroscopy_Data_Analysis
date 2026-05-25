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

#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv") # needs freqShift > 8 excluded
spectra, freq, df = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")
#spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")
#spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")

x_vals, y_vals = bp.build_coordinate_arrays(df)

folded_spectra, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)
keep = shift_freq <= 8
normalSpectra = folded_spectra[:, keep]
shift_freq = shift_freq[keep]
normed = bp.normalize_spectra(normalSpectra)
spectraArray = [normalSpectra]
labls = ["raw", "0.5", "1.5"]
sig = 0.5
for i in range(2):
    spectraArray.append(bp.spatial_smooth_spectra(normed, y_vals, x_vals, sigma=sig))
    labls.append(f"smoothed, sigma={sig}")
    sig = round(sig+1, 2)

kArray = []

for i in range(len(spectraArray)):
    eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(spectraArray[i])
    scores = bd.compute_pc_scores(spectraArray[i], eigenvectors, mean_spectrum, n_components=3)
    PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]

    k_range = list(range(2, 11))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scores)
        inertias.append(km.inertia_)
    pca_knee_k = fe.find_knee_point(k_range, inertias)
    print(f"PCA K-Means knee point: k={pca_knee_k}")

    # Cluster with optimal k
    pca_kmeans = KMeans(n_clusters=pca_knee_k, random_state=42, n_init=10)
    pca_cluster_labels = pca_kmeans.fit_predict(scores)

    bplt.plot_3d_colored_scatter(
        PC1arr, PC2arr, PC3arr,
        colors=pca_cluster_labels,
        xlabel="PC1", ylabel="PC2", zlabel="PC3",
        title=labls[i],
        cmap="tab10"
    )
    kArray.append(pca_cluster_labels)



bplt.plot_component_maps(
    arrays=kArray,
    x_vals=x_vals, y_vals=y_vals,
    labels=labls,
    suptitle="K-means clustering effect of smoothing",
    cmap="tab10", colorbar_label="Cluster",
)