import numpy as np




def run_pca(data):
    """
    Compute the top principal components of the dataset.

    Parameters:
        data : ndarray (N, B) — N spectra, B frequency bins

    Returns:
        eigenvectors : ndarray (B, B) — columns are PCs, sorted by variance (descending)
        eigenvalues  : ndarray (B,)   — corresponding explained variances
        mean_spectrum: ndarray (B,)   — mean spectrum subtracted during centering
    """
    mean_spectrum = np.mean(data, axis=0)
    X = data - mean_spectrum
    cov_matrix = (X.T @ X) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors, eigenvalues, mean_spectrum


def compute_pc_scores(data, eigenvectors, mean_spectrum, n_components=3):
    """
    Project mean-centered spectra onto the top n principal components.

    Parameters:
        data          : ndarray (N, B)
        eigenvectors  : ndarray (B, B) — from run_pca
        mean_spectrum : ndarray (B,)   — from run_pca, subtracted before projecting
        n_components  : int

    Returns:
        scores : ndarray (N, n_components)
    """
    pcs = eigenvectors[:, :n_components]          # (B, n_components)
    return (data - mean_spectrum) @ pcs           # (N, n_components)


def vca(data, num_endmembers, seed=42):
    """
    Vertex Component Analysis (Nascimento & Bioucas-Dias, 2005).
    Extracts endmember spectra from mixed hyperspectral data.

    Parameters:
        data           : ndarray (N, B) — N spectra, B frequency bins
        num_endmembers : int            — number of endmembers to extract
        seed           : int            — random seed for reproducibility

    Returns:
        endmembers : ndarray (num_endmembers, B)
        indices    : ndarray (num_endmembers,) — source pixel of each endmember
    """
    rng = np.random.default_rng(seed)
    N, B = data.shape
    p    = num_endmembers

    mean_spectrum = np.mean(data, axis=0)
    Y_centered    = data - mean_spectrum

    U, S, Vt      = np.linalg.svd(Y_centered, full_matrices=False)
    V_reduced     = Vt[:p - 1, :]                 # (p-1, B)

    Y_proj = Y_centered @ V_reduced.T             # (N, p-1)
    Y_aug  = np.vstack([Y_proj.T, np.ones(N)])    # (p, N)

    A       = np.zeros((p, p))
    A[p-1, 0] = 1
    indices = np.zeros(p, dtype=int)

    for i in range(p):
        w = rng.standard_normal(p)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.linalg.norm(f)
        projections = f @ Y_aug
        k = np.argmax(np.abs(projections))
        A[:, i]  = Y_aug[:, k]
        indices[i] = k

    endmembers = data[indices, :]
    return endmembers, indices


def estimate_abundances(data, endmembers):
    """
    Estimate per-pixel endmember abundances via constrained least squares.

    Abundances are clipped to [0, 1] and normalised to sum to 1 per pixel.

    Parameters:
        data       : ndarray (N, B)
        endmembers : ndarray (p, B)

    Returns:
        abundances : ndarray (N, p)
    """
    A, _, _, _ = np.linalg.lstsq(endmembers.T, data.T, rcond=None)
    abundances  = A.T                              # (N, p)
    abundances  = np.clip(abundances, 0, None)
    row_sums    = abundances.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    abundances  = abundances / row_sums
    return abundances


def compute_reconstruction_error(data, endmembers, abundances):
    """
    Compute per-pixel L2 reconstruction error: ||spectrum - E^T @ a||_2

    Parameters:
        data       : ndarray (N, B)
        endmembers : ndarray (p, B)
        abundances : ndarray (N, p)

    Returns:
        errors : ndarray (N,) — per-pixel reconstruction error
    """
    reconstructed = abundances @ endmembers       # (N, B)
    errors = np.linalg.norm(data - reconstructed, axis=1)
    return errors

