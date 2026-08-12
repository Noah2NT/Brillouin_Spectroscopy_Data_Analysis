"""
Compare derivative kernels for the spectral-gradient mode, on paclitaxel.

The pipeline currently uses one kernel, Sobel [1, 2, 0, -2, -1], chosen because
its 5-pixel span smoothed noise where narrower kernels amplified it. This file
tests that choice against three alternatives of differing width.

All are antisymmetric, so all are valid first-derivative operators:

    [1, 0, -1]                        3 taps, plain central difference
    [1, 2, 0, -2, -1]                 5 taps, Sobel-Feldman (current default)
    [-1, -2, -1, 0, 1, 2, 1]          7 taps, flat-ish weighting
    [1, 2, 3, 0, -3, -2, -1]          7 taps, ramped weighting
    [1, 2, 4, 8, 0, -8, -4, -2, -1]   9 taps, doubling weights
    [1, 2, 3, 4, 0, -4, -3, -2, -1]   9 taps, linear ramp

Normalisation
-------------
The kernels have different gains - their positive weights sum to 1, 3, 4, 6, 15
and 10 - so raw responses are not comparable. Each is divided by the sum of its
positive weights, which makes an ideal step of height h produce a response of h
whichever kernel is used. Set NORMALISE = False to see the raw magnitudes.

What "better" means here
------------------------
Three things are measured, and they do not always agree:

  contrast    mean response at the cell EDGE divided by the response in the
              MEDIA. The media is uniform, so its response is pure noise -
              this is the closest thing to an SNR for an edge detector. A
              kernel scoring ~1 is not detecting the boundary at all.
  texture     mean response inside the spheroid (the 'centre' region), which
              is what the pipeline's gradient mode actually plots.
  separation  eta^2 and a permutation p for the centre-region mean across the
              dose groups - does the kernel distinguish the doses.

Run from the repo root (the data paths are relative).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import carboplatin_line_scans as C
import line_scan_feature_analysis as A


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Where the comparison figures are written. None = show only.
FIG_PATH = None

NORMALISE = True
N_PERM = 20000
SEED = 0

KERNELS = {
    "[1, 0, -1]":                       [1, 0, -1],
    "[1, 2, 0, -2, -1]":                [1, 2, 0, -2, -1],
    "[-1, -2, -1, 0, 1, 2, 1]":         [-1, -2, -1, 0, 1, 2, 1],
    "[1, 2, 3, 0, -3, -2, -1]":         [1, 2, 3, 0, -3, -2, -1],
    "[1, 2, 4, 8, 0, -8, -4, -2, -1]":  [1, 2, 4, 8, 0, -8, -4, -2, -1],
    "[1, 2, 3, 4, 0, -4, -3, -2, -1]":  [1, 2, 3, 4, 0, -4, -3, -2, -1],
}

DRUG_CFG = [d for d in A.DRUGS if d["drug"] == "Paclitaxel"][0]


# ─────────────────────────────────────────────
# Gradient
# ─────────────────────────────────────────────

def spectral_gradient(spectra, kernel, normalise=NORMALISE):
    """Apply a 1D kernel ACROSS pixels; |summed response| per pixel.

    Generalises carboplatin_line_scans.spectral_gradient_per_pixel, which
    hard-codes three kernels. spectra is (n_pixels, n_bins); the kernel runs
    along the pixel axis and the absolute response is summed over every bin,
    so the result is one number per scan position. NaN where the kernel does
    not fit at the ends.
    """
    k = np.asarray(kernel, dtype=float)
    if normalise:
        pos = k[k > 0].sum()
        if pos > 0:
            k = k / pos
    n, half = spectra.shape[0], len(k) // 2
    out = np.full(n, np.nan)
    for i in range(half, n - half):
        window = spectra[i - half:i + half + 1]      # (len(k), n_bins)
        out[i] = np.sum(np.abs(np.tensordot(k, window, axes=(0, 0))))
    return out


def region_means(sig, region):
    """(media, edge, centre) mean response, NaN where a region is empty."""
    out = []
    for name in ("outside", "edge", "center"):
        m = (region == name) & np.isfinite(sig)
        out.append(float(np.mean(sig[m])) if m.any() else np.nan)
    return out


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────

def evaluate(all_results, kernel):
    """Attach this kernel's gradient to every scan and score it.

    Writes two keys per scan: '_grad_full' (whole profile, used for the region
    contrast) and '_grad' (centre-only, matching what the pipeline plots and
    what the dose statistics use).
    """
    rows = []
    for folder in DRUG_CFG["folders"]:
        for r in all_results[folder]:
            spec = r["scan"]["folded_spectra_norm"]
            g = spectral_gradient(spec, kernel)
            r["_grad_full"] = g
            region = r["region"]
            if region is None:
                r["_grad"] = g
                continue
            centre_only = g.copy()
            centre_only[region != "center"] = np.nan
            r["_grad"] = centre_only
            rows.append(region_means(g, region))
    arr = np.array(rows, dtype=float)
    media, edge, centre = np.nanmean(arr, axis=0)
    return dict(media=media, edge=edge, centre=centre,
                contrast=edge / media if media else np.nan)


def dose_separation(all_results, rng):
    """eta^2 / p / rho for the centre-region mean of '_grad' across doses."""
    groups, dose_of = [], []
    for folder, dose in zip(DRUG_CFG["folders"], DRUG_CFG["doses"]):
        by_file = {}
        for r in all_results[folder]:
            v = np.asarray(r["_grad"], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                by_file.setdefault(r["scan"]["filename"], []).append(v.mean())
        g = np.array([np.mean(x) for x in by_file.values()])
        if g.size:
            groups.append(g)
            dose_of += [dose] * g.size
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    from scipy import stats
    rho, _ = stats.spearmanr(dose_of, np.concatenate(groups))
    return (A.eta_squared(groups),
            A.permutation_p(groups, rng, n_perm=N_PERM), rho)


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────

def _save(fig, name):
    if not FIG_PATH:
        return
    os.makedirs(FIG_PATH, exist_ok=True)
    fig.savefig(os.path.join(FIG_PATH, name + ".png"),
                dpi=300, bbox_inches="tight")


def plot_average_lines(all_results, scored):
    """One panel per kernel: the dose-averaged centre-region gradient."""
    names = list(KERNELS)
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.6, nrows * 4.8), squeeze=False)
    axes = axes.flatten()
    handles, labels = [], []
    for ax, name in zip(axes, names):
        evaluate(all_results, KERNELS[name])          # refresh '_grad'
        for folder in DRUG_CFG["folders"]:
            grid, mean, _, _, n = C._aligned_mean(all_results[folder], "_grad")
            if grid is None:
                continue
            ax.plot(grid, mean, 'o-', color=DRUG_CFG["colors"][folder],
                    markersize=3.5, linewidth=1.4,
                    label=f"{folder} (n={n})")
        ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(C.POS_LABEL)
        ax.set_ylabel("|spectral gradient|")
        ax.grid(alpha=0.3)
        C._style_axes(ax)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Paclitaxel - derivative kernel comparison "
                 "(centre region, normalised gain)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if handles:
        fig.legend(handles, labels, title="dose", loc="lower center",
                   ncol=5, frameon=True)
    _save(fig, "kernel_average_lines")
    plt.show()


def plot_region_contrast(scored):
    """Media / edge / centre response per kernel, and the contrast ratio."""
    names = list(KERNELS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    width = 0.26
    for off, key, colour in ((-width, "media", "0.6"),
                             (0.0, "edge", "tab:red"),
                             (width, "centre", "tab:blue")):
        axes[0].bar(x + off, [scored[n][key] for n in names], width,
                    color=colour, label=key)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("mean |spectral gradient|")
    axes[0].set_title("Response by region", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, [scored[n]["contrast"] for n in names], 0.55,
                color="tab:green")
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    axes[1].annotate("1.0 = no edge detected", xy=(0.02, 0.92),
                     xycoords="axes fraction", fontsize=10, color="red",
                     fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("edge / media response")
    axes[1].set_title("Edge contrast (higher is better)", fontsize=13)
    axes[1].grid(alpha=0.3, axis="y")

    for ax in axes:
        C._style_axes(ax)
    fig.suptitle("Paclitaxel - which kernel actually finds the boundary?",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "kernel_region_contrast")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    C.DRUG, C.BASE = DRUG_CFG["drug"], DRUG_CFG["base"]
    C.FOLDERS, C.FOLDER_COLORS = DRUG_CFG["folders"], DRUG_CFG["colors"]
    C.FOLDER_LABELS = DRUG_CFG["labels"]
    C.FOLDER_PATHS = DRUG_CFG.get("paths", {})
    all_results = {f: C.load_and_analyse(C.folder_path(f))
                   for f in DRUG_CFG["folders"]}

    scored = {}
    print(f"\n{'=' * 92}")
    print(f"Paclitaxel - derivative kernel comparison"
          f"{'  (gains normalised)' if NORMALISE else '  (raw gains)'}")
    print("=" * 92)
    print(f"{'kernel':<34}{'media':>9}{'edge':>9}{'centre':>9}"
          f"{'contrast':>10}{'eta^2':>8}{'p_perm':>9}{'rho':>7}")
    print("-" * 92)
    for name, kernel in KERNELS.items():
        s = evaluate(all_results, kernel)
        e2, p, rho = dose_separation(all_results, rng)
        s.update(eta2=e2, p_perm=p, rho=rho)
        scored[name] = s
        print(f"{name:<34}{s['media']:>9.4f}{s['edge']:>9.4f}"
              f"{s['centre']:>9.4f}{s['contrast']:>10.2f}"
              f"{e2:>8.3f}{p:>9.4f}{rho:>7.2f}")
    print("-" * 92)
    print("  contrast = edge/media response; the media is uniform, so its")
    print("  response is noise. A kernel near 1.0 is not seeing the boundary.")

    plot_average_lines(all_results, scored)
    plot_region_contrast(scored)
