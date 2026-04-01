import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import curvesMapping as cm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator


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


    
# ===============
# PC Computation
# ===============

PC1, PC2, PC3 = cm.compute_pcs(peakSpectra)

PC1List = []
PC2List = []
PC3List = []

for spectrum in peakSpectra:
    PC1List.append(np.dot(spectrum, PC1))
    PC2List.append(np.dot(spectrum, PC2))
    PC3List.append(np.dot(spectrum, PC3))
PC1arr = np.array(PC1List)
PC2arr = np.array(PC2List)
PC3arr = np.array(PC3List)


# ============================
# PCA Abundance Heatmaps
# ============================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
PC_labels = ['PC1', 'PC2', 'PC3']
PC_arrays = [PC1arr, PC2arr, PC3arr]

for i in range(3):
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
# k-means Clustering (bad)
# ===================

'''
X = np.column_stack((PC1List, PC2List, PC3List))
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='tab10', s=20, alpha=0.3)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=2000, c='red', marker='X', label='Centroids')
ax.set_title('K-means clustering in PC space')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
ax.view_init(elev=90, azim=210)
plt.show()

cluster_map = labels.reshape(len(y_vals), len(x_vals))
plt.figure(figsize=(6, 5))
plt.imshow(cluster_map, origin="lower", cmap="tab10")
plt.colorbar(label='Cluster')
plt.title('Cluster Heatmap')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''


# ===================
# Compute Peak Shift
# ===================

freqShift = [] # intensity, frequency shift, peak width, C
for spectrum in peakSpectra:

    mean = peakFreq[np.argmax(spectrum)]
    C = -13
    amp = spectrum.max()-C
    gamma = 0.36
    params = np.array([amp, mean, gamma, C], dtype=float)
    steps = np.array([0.01*amp, 0.01, 0.01*gamma, 0.1], dtype=float)
    phase = 0
    for iteration in range(50):
        params_prev = params.copy()
        params, err = cm.update_parameter(peakFreq, spectrum, params, 1, steps[1])  # mean
        if(np.array_equal(params, params_prev)):
            if(phase < 4):
                phase = phase + 1
                steps *= 0.5
            else:
                break
    freqShift.append(params[1])
freqShiftarr = np.array(freqShift)


# ===================
# Plot according to peak shift in PC space (good) (noticable gradient)
# ===================


i = 0 # plots in 3d from all angles
while i < 360:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(PC1List, PC2List, PC3List, c=freqShift, cmap='magma', s=10)
    
    plt.colorbar(sc, label='Frequency shift')
    
    ax.set_title(str(i))
    ax.set_xlabel('PC1 value')
    ax.set_ylabel('PC2 value')
    ax.set_zlabel('PC3 value')
    ax.view_init(elev=0, azim=i)
    plt.show()
    i += 10


fig = plt.figure() # plots in 3d at the clearest angle
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(PC1List, PC2List, PC3List, c=freqShift, cmap='magma', s=10)
plt.colorbar(sc, label='Frequency Shift (GHz)')
ax.set_title("Brillouin data in PC space")
ax.set_xlabel('PC1 value')
ax.set_ylabel('PC2 value')
ax.set_zlabel('PC3 value')
ax.view_init(elev=0, azim=210)
plt.show()


plt.figure() # plots in 2d
sc = plt.scatter(PC1List, PC2List, c=freqShift, cmap='magma', s=20)
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
