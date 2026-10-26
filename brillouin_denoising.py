"""Spatial-spectral reconstruction of raw Poisson count cubes (Y, X, bins).

Salmon et al., arXiv:1206.0338v4, equations 10, 13, 14 and Algorithm 1.
This implements the non-sparse NLPCA variant, adapted to full-spectrum spatial
patches. It estimates expected counts, NOT material-pure endmembers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.special import xlogy
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class DenoiseConfig:
    method: str = "poisson_nlpca"
    rank: int = 3
    patch: int = 3
    clusters: int = 4
    iterations: int = 20
    sigma: float = 0.8
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def validate_counts(counts):
    y = np.asarray(counts, dtype=float)
    if y.ndim != 3 or min(y.shape) < 1:
        raise ValueError("Expected a nonempty (height, width, spectral bins) cube")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Counts must be finite and nonnegative; do not subtract baseline")
    return y


def poisson_deviance(y, mean):
    """Elementwise deviance, including finite predictions for zero counts."""
    y, mean = np.broadcast_arrays(np.asarray(y, float), np.asarray(mean, float))
    mean = np.maximum(mean, 1e-12)
    return 2 * (xlogy(y, y / mean) - y + mean)


def thin_counts(counts, fraction, seed=0):
    """Independent Poisson splits under the Poisson model, before preprocessing."""
    y = validate_counts(counts)
    if not 0 < fraction < 1 or np.any(y != np.floor(y)):
        raise ValueError("Thinning needs integer photon counts and 0 < fraction < 1")
    a = np.random.default_rng(seed).binomial(y.astype(np.int64), fraction)
    return a, y.astype(np.int64) - a


def extract_patches(cube, size):
    """Valid spatial patches, full spectrum; no reflected duplicate observations."""
    h, w, b = cube.shape
    if size < 1 or size > min(h, w):
        raise ValueError("Patch size must fit the spatial dimensions")
    blocks = [cube[dy:h-size+dy+1, dx:w-size+dx+1]
              for dy in range(size) for dx in range(size)]
    return np.stack(blocks, axis=2).reshape(-1, size * size * b)


def reproject_patches(patches, shape, size):
    h, w, b = shape
    grid = patches.reshape(h-size+1, w-size+1, size*size, b)
    out, weights = np.zeros(shape), np.zeros((h, w, 1))
    for k, (dy, dx) in enumerate((dy, dx) for dy in range(size) for dx in range(size)):
        out[dy:h-size+dy+1, dx:w-size+dx+1] += grid[:, :, k]
        weights[dy:h-size+dy+1, dx:w-size+dx+1] += 1
    return out / weights


def poisson_clusters(features, clusters, seed=0, iterations=30):
    """Bregman hard clustering (paper Algorithm 2), arithmetic-mean centres."""
    rng = np.random.default_rng(seed)
    k = min(clusters, len(features))
    centres = np.maximum(features[rng.choice(len(features), k, replace=False)], 1e-6)
    previous = None
    for _ in range(iterations):
        distance = centres.sum(axis=1)[None] - features @ np.log(centres).T
        labels = distance.argmin(axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for c in range(k):
            members = features[labels == c]
            centres[c] = (members.mean(axis=0) if len(members)
                          else features[rng.integers(len(features))])
        centres = np.maximum(centres, 1e-6)
    return labels


def _svd(matrix, rank, seed):
    rank = min(rank, *matrix.shape)
    return randomized_svd(matrix, n_components=rank, n_iter=3, random_state=seed)


def poisson_low_rank(y, rank=3, iterations=20, seed=0, tolerance=1e-5, callback=None):
    """Fit exp(UV) with alternating full block-Newton updates and line searches.

    Hessians contain cross-component terms (V diag(mu) V.T), not merely their
    diagonal entries. Ridge damping stabilizes steps, not the final objective.
    Log-smoothed SVD initialization replaces the paper's random initialization.
    The actual objective always uses unsmoothed counts. Loss history is returned
    for numerical checks; overlap means it is a patch composite likelihood.
    """
    y = np.asarray(y, float)
    if y.ndim != 2 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Expected a finite nonnegative matrix")
    if rank < 1 or iterations < 1:
        raise ValueError("Rank and iterations must be positive")
    if not np.any(y):
        if callback is not None:
            callback("initial", np.zeros_like(y), 0.0)
        return np.zeros_like(y), [0.0]
    pilot = 0.65 * y + 0.35 * y.mean(axis=0, keepdims=True)
    u, s, v = _svd(np.log(pilot + 0.2), rank, seed)
    u = u * s
    # Exponent bounds are numerical guards, enforced by line search, not by
    # differentiating a clipped exponential with an incorrect derivative.
    def loss(a, b):
        eta = a @ b
        if np.max(eta) > 40 or np.min(eta) < -40:
            return np.inf
        return float(np.sum(np.exp(eta) - y * eta))
    history = [loss(u, v)]
    if callback is not None:
        callback("initial", np.exp(u @ v), history[0])
    for iteration in range(iterations):
        old_mu = np.exp(u @ v)
        for update_u in (True, False):
            mu = np.exp(u @ v)
            if update_u:
                grad = (mu - y) @ v.T
                hess = np.einsum('ij,aj,bj->iab', mu, v, v, optimize=True)
            else:
                grad = ((mu - y).T @ u)
                hess = np.einsum('ij,ia,ib->jab', mu, u, u, optimize=True)
            ridge = 1e-6 * np.maximum(np.trace(hess, axis1=1, axis2=2) / hess.shape[-1], 1)
            hess += ridge[:, None, None] * np.eye(hess.shape[-1])[None]
            step = np.linalg.solve(hess, grad[..., None])[..., 0]
            before = loss(u, v)
            scale = 1.0
            for _ in range(24):
                un, vn = (u-scale*step, v) if update_u else (u, v-scale*step.T)
                if loss(un, vn) <= before:
                    u, v = un, vn
                    break
                scale *= 0.5
            if callback is not None:
                callback(f"iteration {iteration+1}: {'coefficients U' if update_u else 'dictionary V'}", np.exp(u @ v), loss(u, v))
        # Gauge balancing preserves UV exactly and improves conditioning.
        norms = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
        v = v / norms[:, None]
        u = u * norms
        history.append(loss(u, v))
        change = np.linalg.norm(np.exp(u @ v) - old_mu) / max(np.linalg.norm(old_mu), 1e-12)
        if change < tolerance:
            break
    return np.exp(u @ v), history


def reconstruct(counts, config=None, trace=None):
    """Return estimated expected counts with the input shape and exposure.

    Optional trace is a mutable dictionary populated with every Newton half-step,
    per-group predictions, aligned reprojected cubes and the final patch proposals.
    It can use substantial memory; omit it for normal production reconstruction.
    No truth, masks, material labels or high-exposure reference enter this API.
    Input is a complete rectangular raster, already restricted to a meaningful
    spectral window. Spectral bins must remain in their original count units.
    """
    y = validate_counts(counts)
    c = config or DenoiseConfig()
    if c.rank < 1 or c.clusters < 1 or c.iterations < 1 or c.sigma < 0:
        raise ValueError("Invalid rank, cluster count, iteration count or sigma")
    if c.method == "raw":
        return y.copy()
    if c.method == "spatial_mean":
        return uniform_filter(y, size=(c.patch, c.patch, 1), mode="reflect")
    if c.method == "gaussian":
        return gaussian_filter(y, sigma=(c.sigma, c.sigma, 0), mode="reflect")
    if c.method == "global_pca":
        flat = y.reshape(-1, y.shape[-1])
        mean = flat.mean(axis=0)
        u, s, v = _svd(flat - mean, c.rank, c.seed)
        return np.maximum((u*s) @ v + mean, 0).reshape(y.shape)
    if c.method not in {"poisson_nlpca", "poisson_spectral", "anscombe_nlpca"}:
        raise ValueError(f"Unknown reconstruction method: {c.method}")
    if c.method == "poisson_spectral" and c.patch != 1:
        raise ValueError("poisson_spectral uses one spatial pixel per spectral patch")
    patches = extract_patches(y, c.patch)
    # Sum consecutive bands for noise-resistant clustering, while retaining
    # coarse spectral shape. Unlike spectral-intensity-only clustering in the
    # paper's 3D experiment, this distinguishes equally bright Brillouin peaks.
    indices = np.array_split(np.arange(y.shape[-1]), min(16, y.shape[-1]))
    coarse = np.stack([y[:, :, ix].sum(axis=2) for ix in indices], axis=2)
    labels = poisson_clusters(extract_patches(coarse, c.patch), c.clusters, c.seed)
    if trace is not None:
        if c.method not in {"poisson_nlpca", "poisson_spectral"}:
            raise ValueError("Tracing requires a Poisson method")
        trace.clear()
        trace.update(labels=labels.copy(), patch_shape=patches.shape, groups={})
    result = np.empty_like(patches)
    for cluster in np.unique(labels):
        mask = labels == cluster
        group = patches[mask]
        if c.method in {"poisson_nlpca", "poisson_spectral"}:
            records = []
            def record(name, prediction, objective):
                records.append((name, prediction.copy(), objective))
            result[mask], _ = poisson_low_rank(group, c.rank, c.iterations, c.seed,
                                              callback=record if trace is not None else None)
            if trace is not None:
                trace["groups"][int(cluster)] = records
        else:
            z = 2 * np.sqrt(group + 3/8)
            mean = z.mean(axis=0)
            u, s, v = _svd(z - mean, c.rank, c.seed)
            zhat = np.maximum((u*s) @ v + mean, 1)
            # Standard asymptotically unbiased inverse (not the naive inverse).
            result[mask] = np.maximum(zhat**2/4 - 1/8 + np.sqrt(3/2)/(4*zhat)
                                     - 11/(8*zhat**2) + 5*np.sqrt(3/2)/(8*zhat**3), 0)
    if trace is not None:
        # Align independent cluster updates by their local iteration number.
        # Converged groups retain their last prediction. This is not the order
        # of wall-clock execution of the sequential cluster loop.
        stages = max(len(records) for records in trace["groups"].values())
        trace["names"], trace["cubes"], trace["losses"] = [], [], []
        longest = max(trace["groups"].values(), key=len)
        for step in range(stages):
            stage_patches = np.empty_like(patches)
            objective = 0.0
            for cluster, records in trace["groups"].items():
                _, prediction, value = records[min(step, len(records)-1)]
                stage_patches[labels == cluster] = prediction
                objective += value
            trace["names"].append(longest[step][0])
            trace["cubes"].append(reproject_patches(stage_patches, y.shape, c.patch))
            trace["losses"].append(objective)
        trace["final_patches"] = result.copy()
    return reproject_patches(result, y.shape, c.patch)
