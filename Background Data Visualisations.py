import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt
import feature_extraction as fe

'''
1. Fold Stokes / anti-Stokes to get a single clean Brillouin peak
2. Run PCA → determine n_endmembers (95% variance)
3. Run VCA → extract endmembers and abundances
4. Fit single Lorentzian → extract Brillouin frequency shift, amplitude, linewidth
5. Feature extraction on each endmember abundance map AND frequency shift map
6. K-means clustering on each endmember with elbow method
7. Display all transformations side-by-side
'''

# ===================
# Data Loading
# ===================

#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv") # needs freqShift > 8 excluded
#spectra, freq, df = bp.load_brillouin_csv("XY_scanning_40um_60um_step1um_10s_fixed.csv")
spectra, freq, df = bp.load_brillouin_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv")
#spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv")

x_vals, y_vals = bp.build_coordinate_arrays(df)

# ===================
# Stokes / Anti-Stokes Folding
# ===================

folded_spectra, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)

keep = shift_freq <= 8
folded_spectra = folded_spectra[:, keep]
shift_freq = shift_freq[keep]

print(f"Folded spectra shape: {folded_spectra.shape}")
print(f"Shift freq range: {shift_freq[0]:.3f} — {shift_freq[-1]:.3f} GHz")

# ========================
# Find Relevant Components
# ========================

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(folded_spectra)
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance      = np.cumsum(explained_variance_ratio)
n_endmembers             = int(np.searchsorted(cumulative_variance, 0.95) + 1)
n_endmembers = 3 if n_endmembers == 2 else n_endmembers
n_endmembers = 2 if n_endmembers == 1 else n_endmembers
print(f"Components to explain 95% variance: {n_endmembers}")
print(f"Cumulative variance at {n_endmembers} components: {cumulative_variance[n_endmembers - 1]:.4f}")

# ===================================
# K-Means Clustering in PCA Space
# ===================================

scores = bd.compute_pc_scores(folded_spectra, eigenvectors, mean_spectrum, n_components=3)
PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]

# Elbow method on PC scores to find optimal k
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
    title=f"PCA Space — K-Means (k={pca_knee_k})",
    cmap="tab10",
)

# ===================================
# PCA Score Maps + Cluster Heatmap
# ===================================

pca_cluster_map = bp.reshape_to_map(pca_cluster_labels, y_vals, x_vals)

pc_score_maps = []
for c in range(3):
    pc_map = bp.reshape_to_map(scores[:, c], y_vals, x_vals)
    pc_score_maps.append(pc_map)

# ========================
# VCA Endmember Extraction
# ========================

endmembers, endmember_indices = bd.vca(folded_spectra, n_endmembers, seed=69)
abundances = bd.estimate_abundances(folded_spectra, endmembers)
endmember_pos = []
for i, idx in enumerate(endmember_indices):
    row = df.iloc[idx]
    endmember_pos.append((row['X'], row['Y']))

# ===================================
# 3D PCA Scatter with VCA Endmembers
# ===================================

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

# ==============================
# Brillouin Frequency Shift Fit
# ==============================

fitted_curves, fit_params = bf.fit_all_spectra(shift_freq, folded_spectra)
amplitudes, peak_shifts, linewidths, baselines = bf.extract_fit_parameters(fit_params)

shift_map = bp.reshape_to_map(peak_shifts, y_vals, x_vals)
amp_map   = bp.reshape_to_map(amplitudes, y_vals, x_vals)
width_map = bp.reshape_to_map(linewidths, y_vals, x_vals)

print(f"Brillouin shift range: {peak_shifts.min():.3f} — {peak_shifts.max():.3f} GHz")
print(f"Mean shift: {peak_shifts.mean():.3f} GHz")

# ==============================
# Summary Grid: Fitting + VCA
# ==============================

summary_panels = [
    {'data': shift_map, 'title': 'Brillouin Shift (GHz)', 'cmap': 'magma', 'colorbar_label': 'GHz'},
    {'data': amp_map,   'title': 'Peak Amplitude',        'cmap': 'inferno', 'colorbar_label': 'Intensity'},
    {'data': width_map, 'title': 'Linewidth',             'cmap': 'viridis', 'colorbar_label': 'GHz'},
] + [
    {'data': m, 'title': f'PC{i+1} Scores', 'cmap': 'RdBu', 'colorbar_label': 'Score'}
    for i, m in enumerate(pc_score_maps)
] + [
    {'data': pca_cluster_map, 'title': f'PCA K-Means (k={pca_knee_k})', 'cmap': 'tab10', 'colorbar_label': 'Cluster'},
] + [
    {'data': m, 'title': f'Endmember {i+1}', 'cmap': 'inferno', 'colorbar_label': 'Abundance'}
    for i, m in enumerate(abundance_maps)
]
bplt.plot_analysis_grid(summary_panels, suptitle="Brillouin Analysis Summary")

# ==========================================
# Feature Extraction (all at the end)
# ==========================================

for i in range(n_endmembers):
    fe.extract_features_grid(abundance_maps[i], f"Endmember {i+1}")

for i, pc_map in enumerate(pc_score_maps):
    fe.extract_features_grid(pc_map, f"PC{i+1} Scores")

fe.extract_features_grid(shift_map, "Brillouin Frequency Shift")
