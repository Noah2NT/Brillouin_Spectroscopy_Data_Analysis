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


def _non_maximum_suppression(magnitude, direction):
    """Thin a gradient image to local maxima along the gradient direction."""
    magnitude = np.asarray(magnitude, dtype=float)
    angle = (np.rad2deg(direction) + 180.0) % 180.0
    thinned = np.zeros_like(magnitude)

    centre = magnitude[1:-1, 1:-1]
    horizontal = ((angle[1:-1, 1:-1] < 22.5) |
                  (angle[1:-1, 1:-1] >= 157.5))
    diagonal_up = ((angle[1:-1, 1:-1] >= 22.5) &
                   (angle[1:-1, 1:-1] < 67.5))
    vertical = ((angle[1:-1, 1:-1] >= 67.5) &
                (angle[1:-1, 1:-1] < 112.5))
    diagonal_down = ((angle[1:-1, 1:-1] >= 112.5) &
                     (angle[1:-1, 1:-1] < 157.5))

    keep = np.zeros_like(centre, dtype=bool)
    keep |= (horizontal & (centre >= magnitude[1:-1, :-2]) &
             (centre >= magnitude[1:-1, 2:]))
    keep |= (diagonal_up & (centre >= magnitude[:-2, 2:]) &
             (centre >= magnitude[2:, :-2]))
    keep |= (vertical & (centre >= magnitude[:-2, 1:-1]) &
             (centre >= magnitude[2:, 1:-1]))
    keep |= (diagonal_down & (centre >= magnitude[:-2, :-2]) &
             (centre >= magnitude[2:, 2:]))
    thinned[1:-1, 1:-1] = np.where(keep, centre, 0.0)
    return thinned


def canny_edges(arr, sigma=1.0, low_threshold=0.1, high_threshold=0.2,
                return_stages=False):
    """Detect edges in a 2D numeric map using the Canny algorithm.

    The default thresholds are fractions of the non-maximum-suppressed peak.
    Values above 1 are interpreted as absolute gradient magnitudes. NaNs and
    infinities are filled before filtering and never appear as output edges.

    Parameters:
        arr             : 2D array-like input image or spatial map
        sigma           : non-negative Gaussian smoothing standard deviation
        low_threshold   : weak-edge threshold (fraction or absolute value)
        high_threshold  : strong-edge threshold (fraction or absolute value)
        return_stages   : if True, also return every intermediate result

    Returns:
        edges : boolean ndarray
        stages : dict of ndarrays, only when return_stages=True
    """
    image = np.asarray(arr, dtype=float)
    if image.ndim != 2:
        raise ValueError("canny_edges expects a 2D array")
    if image.size == 0 or min(image.shape) < 3:
        raise ValueError("canny_edges expects at least a 3 x 3 array")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not (0 <= low_threshold < high_threshold):
        raise ValueError("thresholds must satisfy 0 <= low < high")

    invalid = ~np.isfinite(image)
    if invalid.all():
        raise ValueError("arr must contain at least one finite value")
    filled = image.copy()
    filled[invalid] = np.nanmedian(filled)

    smoothed = gaussian_filter(filled, sigma=sigma, mode="reflect")
    gradient_x = sobel(smoothed, axis=1, mode="reflect")
    gradient_y = sobel(smoothed, axis=0, mode="reflect")
    magnitude = np.hypot(gradient_x, gradient_y)
    direction = np.arctan2(gradient_y, gradient_x)
    suppressed = _non_maximum_suppression(magnitude, direction)

    peak = float(suppressed.max())
    low = low_threshold * peak if low_threshold <= 1 else low_threshold
    high = high_threshold * peak if high_threshold <= 1 else high_threshold
    strong = suppressed >= high if peak > 0 else np.zeros_like(image, dtype=bool)
    weak = ((suppressed > 0) & (suppressed >= low) & ~strong
            if peak > 0 else strong.copy())

    # Hysteresis: retain weak pixels connected to strong pixels (8-neighbourhood).
    edges = strong.copy()
    candidates = strong | weak
    while True:
        grown = binary_dilation(edges, structure=np.ones((3, 3))) & candidates
        if np.array_equal(grown, edges):
            break
        edges = grown
    edges[invalid] = False

    if not return_stages:
        return edges

    stages = {
        "input": image.copy(),
        "smoothed": smoothed,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "gradient_magnitude": magnitude,
        "gradient_direction": direction,
        "non_maximum_suppression": suppressed,
        "weak_edges": weak,
        "strong_edges": strong,
        "edges": edges,
    }
    return edges, stages


def plot_canny_stages(stages, figsize=(15, 8)):
    """Plot the main Canny stages returned by ``canny_edges``."""
    panels = [
        ("input", "1. Input", "gray"),
        ("smoothed", "2. Gaussian smoothing", "gray"),
        ("gradient_magnitude", "3. Sobel magnitude", "magma"),
        ("gradient_direction", "4. Gradient direction", "twilight"),
        ("non_maximum_suppression", "5. Non-maximum suppression", "magma"),
        ("strong_edges", "6. Strong edges", "gray"),
        ("weak_edges", "7. Weak edges", "gray"),
        ("edges", "8. Hysteresis result", "gray"),
    ]
    missing = [key for key, _, _ in panels if key not in stages]
    if missing:
        raise ValueError(f"missing Canny stages: {', '.join(missing)}")

    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes.flat, panels):
        image = ax.imshow(stages[key], origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if key in {"gradient_magnitude", "gradient_direction",
                   "non_maximum_suppression"}:
            fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Canny edge detection, step by step")
    return fig, axes


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
