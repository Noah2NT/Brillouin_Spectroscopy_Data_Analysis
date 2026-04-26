import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd
import brillouin_plotting as bplt
import feature_extraction as fe

def compute_variance(values1, values2): # 2 1D arrays with equal length
    variance = 0
    for i in range(len(values1)):
        variance += (values2[i] - values1[i])**2
    return np.sqrt(variance)

def num_neighbors(dataset, index, distance):
    root = dataset[index]
    n = 0
    for d in dataset:
        if compute_variance(root, d) < distance:
            n += 1
    return n - 1

def in_range(value1, value2, distance):
    if compute_variance(value1, value2) < distance:
        return True
    return False

spectra, freq, df = bp.load_brillouin_csv("water_plexiglass.csv") # 6 neighbors, dist 500
#spectra, freq, df = bp.load_brillouin_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv") # 11 neighbors, dist 60

x_vals, y_vals = bp.build_coordinate_arrays(df)
spectra = spectra[:, 300:512]
freq = freq[300:512]

'''
keep = freq <= 8
spectra = spectra[:, keep]
freq = freq[keep]
'''

eigenvectors, eigenvalues, mean_spectrum = bd.run_pca(spectra)
scores = bd.compute_pc_scores(spectra, eigenvectors, mean_spectrum, n_components=3)
PC1arr, PC2arr, PC3arr = scores[:, 0], scores[:, 1], scores[:, 2]
components = scores[:, :3]




distance = 500
neighbors = 6
popular = [-1] * spectra.shape[0]

# plot initial map
bplt.plot_2d_colored_scatter(PC1arr, PC2arr, popular, cmap="viridis")
bplt.plot_3d_colored_scatter(PC1arr, PC2arr, PC3arr, popular, cmap="viridis")


for i in range(components.shape[0]):
    if num_neighbors(components, i, distance) >= neighbors:
        popular[i] = 0

# plot popular datapoints
bplt.plot_2d_colored_scatter(PC1arr, PC2arr, popular, cmap="viridis")
bplt.plot_3d_colored_scatter(PC1arr, PC2arr, PC3arr, popular, cmap="viridis")
bplt.plot_heatmap(popular, x_vals, y_vals, cmap = "viridis")




cluster = 1

matching_indices = [i for i, val in enumerate(popular) if val == 0]
while matching_indices:
    random_index = random.choice(matching_indices)
    queue = [random_index]
    while len(queue) > 0:
        change = False # for visuals
        index = queue.pop(random.randrange(len(queue)))
        for i in range(components.shape[0]):
            if in_range(components[i], components[index], distance):
                if popular[i] == 0:
                    queue.append(i)
                if popular[i] != cluster:
                    popular[i] = cluster
                    change = True # for visuals
        if change:
            bplt.plot_3d_colored_scatter(PC1arr, PC2arr, PC3arr, popular, cmap="viridis")
    cluster += 1
    matching_indices = [i for i, val in enumerate(popular) if val == 0]

for i in range(len(popular)): # assign the rest of the values a cluster
    if popular[i] == -1:
        popular[i] = cluster


#bplt.plot_2d_colored_scatter(PC1arr, PC2arr, popular, cmap="viridis")
bplt.plot_heatmap(popular, x_vals, y_vals, cmap = "viridis")
bplt.plot_3d_colored_scatter(PC1arr, PC2arr, PC3arr, popular, cmap="viridis")
