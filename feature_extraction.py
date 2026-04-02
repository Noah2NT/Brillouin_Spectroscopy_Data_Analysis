import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    laplace,
    maximum_filter,
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing
)

def max_pool(arr, k=2):
    ny, nx = arr.shape
    out = np.zeros((ny // k, nx // k))
    for i in range(0, ny - ny % k, k):
        for j in range(0, nx - nx % k, k):
            out[i // k, j // k] = np.max(arr[i:i+k, j:j+k])
    return out


def avg_pool(arr, k=2):
    ny, nx = arr.shape
    out = np.zeros((ny // k, nx // k))
    for i in range(0, ny - ny % k, k):
        for j in range(0, nx - nx % k, k):
            out[i // k, j // k] = np.mean(arr[i:i+k, j:j+k])
    return out


def smooth_heatmap(arr, sigma=1):
    return gaussian_filter(arr, sigma=sigma)


def gradient_magnitude(arr):
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx**2 + gy**2)
    return grad


def sobel_edges(arr):
    gx = sobel(arr, axis=1)
    gy = sobel(arr, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    return edges


def laplacian_filter(arr):
    return laplace(arr)


def threshold_mask(arr, threshold):
    return (arr > threshold).astype(int)


def local_maxima(arr, size=3):
    max_arr = maximum_filter(arr, size=size)
    return (arr == max_arr).astype(int)


def erode_mask(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        out = binary_erosion(out)
    return out.astype(int)


def dilate_mask(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        out = binary_dilation(out)
    return out.astype(int)


def open_mask(mask):
    return binary_opening(mask.astype(bool)).astype(int)


def close_mask(mask):
    return binary_closing(mask.astype(bool)).astype(int)


def clean_mask(mask):
    mask = binary_opening(mask.astype(bool))
    mask = binary_closing(mask.astype(bool))
    return mask.astype(int)


def find_knee_point(k_range, inertias):
    """
    Find the knee/elbow point using maximum perpendicular distance
    from the line connecting the first and last points.
    """
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    best_dist = -1
    best_idx = 0
    for i in range(len(k_range)):
        pt = np.array([k_range[i], inertias[i]])
        dist = np.abs(line_vec[0]*(p1[1]-pt[1]) - line_vec[1]*(p1[0]-pt[0])) / line_len
        if dist > best_dist:
            best_dist = dist
            best_idx = i

    return k_range[best_idx]


def extract_features_grid(map_2d, label, max_k=10):
    """
    Run full feature extraction + k-means with elbow analysis on a 2D map
    and display as a 12-panel grid.

    Parameters:
        map_2d : ndarray (ny, nx) — the spatial map to analyze
        label  : str — name for the suptitle
        max_k  : int — maximum k to test for elbow method
    """
    from sklearn.cluster import KMeans
    import brillouin_plotting as bplt

    # Pooling
    max_pooled = max_pool(map_2d, k=2)
    avg_pooled = avg_pool(map_2d, k=2)

    # Smoothing
    smoothed = smooth_heatmap(map_2d, sigma=1)

    # Edge / change detection
    grad      = gradient_magnitude(map_2d)
    sobel_map = sobel_edges(map_2d)
    lap_map   = laplacian_filter(map_2d)

    # Thresholding
    threshold = np.mean(map_2d)
    mask      = threshold_mask(map_2d, threshold)

    # Local maxima
    peaks = local_maxima(map_2d, size=3)

    # Morphological cleanup
    cleaned = clean_mask(mask)

    # K-means elbow analysis
    flat = map_2d.flatten().reshape(-1, 1)
    k_range = list(range(2, max_k + 1))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(flat)
        inertias.append(km.inertia_)

    knee_k = find_knee_point(k_range, inertias)
    print(f"  {label}: knee point at k={knee_k}")

    # K-means with optimal k
    kmeans = KMeans(n_clusters=knee_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(flat)
    cluster_map = cluster_labels.reshape(map_2d.shape)

    # Elbow plot function for the grid
    def plot_elbow(ax):
        ax.plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=5)
        ax.axvline(knee_k, color='red', linestyle='--', linewidth=1.5, label=f'Knee k={knee_k}')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.legend(fontsize=9)

    # Side-by-side grid (12 panels = 3 rows x 4 cols)
    panels = [
        {'data': map_2d,     'title': 'Original',       'cmap': 'inferno', 'colorbar_label': 'Value'},
        {'data': smoothed,   'title': 'Smoothed',        'cmap': 'inferno', 'colorbar_label': 'Value'},
        {'data': max_pooled, 'title': 'Max Pooled',      'cmap': 'inferno', 'colorbar_label': 'Value'},
        {'data': avg_pooled, 'title': 'Avg Pooled',      'cmap': 'inferno', 'colorbar_label': 'Value'},
        {'data': grad,       'title': 'Gradient Mag',    'cmap': 'magma',   'colorbar_label': 'Change'},
        {'data': sobel_map,  'title': 'Sobel Edges',     'cmap': 'magma',   'colorbar_label': 'Edge'},
        {'data': lap_map,    'title': 'Laplacian',       'cmap': 'RdBu',    'colorbar_label': 'Curvature'},
        {'data': mask,       'title': 'Threshold Mask',  'cmap': 'gray',    'colorbar_label': 'Binary'},
        {'data': cleaned,    'title': 'Cleaned Mask',    'cmap': 'gray',    'colorbar_label': 'Binary'},
        {'data': peaks,      'title': 'Local Maxima',    'cmap': 'gray',    'colorbar_label': 'Binary'},
        {'data': cluster_map,'title': f'K-Means (k={knee_k})', 'cmap': 'tab10', 'colorbar_label': 'Cluster'},
        {'title': 'Elbow Method',  'plot_func': plot_elbow},
    ]
    bplt.plot_analysis_grid(panels, suptitle=f"{label} — Feature Extraction")
