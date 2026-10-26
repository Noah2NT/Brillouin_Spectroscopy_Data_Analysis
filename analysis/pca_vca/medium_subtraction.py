"""Ternary unmixing (medium / cytoplasm / nucleus), medium removal, PCA scoring.

Builds on background_endmember.py:
  1. Isolate a confident background region and co-add it into a pure medium
     endmember (assumed 100% medium / water).
  2. Anchored VCA(3): medium is pinned as EM1; the two free vertices are
     searched *inside the cell mask only*, so they describe cell material
     rather than a noisy background pixel.
  3. Per-pixel abundances -> an RGB composite (R=nucleus, G=cytoplasm,
     B=medium), i.e. "what % of each pure component is this pixel".
  4. Medium removal at strength k:   r_i = s_i - k * a_i,medium * m
     k = 0 leaves the data untouched; k = 1 removes the entire estimated
     medium contribution. k is swept because the right value is not known
     a priori.
  5. Score each k by PCA on the corrected data: explained variance of
     PC1+PC2. The premise is that once the medium axis is gone the cell is
     a ~2-component system, so 2 PCs should capture most of what is left.

Two guards on that score, because it is confoundable:
  * NULL CONTROL — repeat the subtraction with spatially shuffled abundances.
    A real effect must beat the null. If the null improves just as much, the
    "gain" is just removal of a rank-1 direction, not of the medium.
  * EIGENVALUE GAP (lambda2 / lambda3) — explained variance can fall even when
    the subtraction is correct, because removing a large signal direction
    leaves noise as a bigger *share* of a much smaller total. The gap says
    whether a genuine 2nd component still stands above the noise continuum.
"""

from pathlib import Path
import importlib.util
import sys, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_decomposition as bd

_spec = importlib.util.spec_from_file_location(
    "background_endmember", Path(__file__).with_name("background_endmember.py"))
bge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bge)


DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "medium_subtraction"

K_SWEEP = np.round(np.linspace(0.0, 1.0, 21), 3)
RENORM_FLOOR = 0.05     # skip pixels whose surviving signal is < this fraction
N_NULL = 5              # shuffled-abundance repeats
SEED = 42


# ─────────────────────────────────────────────
# PCA scoring
# ─────────────────────────────────────────────

NOISE_COMPONENTS = slice(3, 10)   # PC4..PC10 taken as the noise continuum


def morans_i(values, yi, xi, shape):
    """4-neighbour spatial autocorrelation of a per-pixel value.

    Scale-free and immune to how much total variance a step removed, so it
    answers the question explained variance cannot: is what survives a
    spatially coherent structure (a nucleus) or salt-and-pepper noise?
    I ~ 0 means no spatial structure; I -> 1 means smooth structure.
    """
    grid = np.full(shape, np.nan)
    grid[yi, xi] = values
    finite = np.isfinite(grid)
    if finite.sum() < 10:
        return np.nan
    dev = grid - np.nanmean(grid)
    denom = np.nansum(dev ** 2)
    if denom <= 0:
        return np.nan

    num, n_pairs = 0.0, 0
    for axis, shift in ((0, 1), (1, 1)):
        a = np.roll(dev, shift, axis=axis)
        valid = finite & np.roll(finite, shift, axis=axis)
        if axis == 0:
            valid[:shift] = False
        else:
            valid[:, :shift] = False
        num += float(np.nansum(dev[valid] * a[valid]))
        n_pairs += int(valid.sum())
    if n_pairs == 0:
        return np.nan
    # Each undirected pair is counted once above; W = 2 * n_pairs.
    return float((finite.sum() / (2 * n_pairs)) * (2 * num) / denom)


def pca_scores(spectra, index=None, yi=None, xi=None, shape=None):
    """PCA diagnostics for a set of spectra.

    evr1/evr2      — explained variance of PC1, PC1+PC2 (the requested metric)
    lam2_over_noise— lambda2 divided by the mean PC4..PC10 eigenvalue. Both are
                     absolute variances, so the ratio is unaffected by how much
                     total variance a subtraction removed. This is the honest
                     version of "is there a real second component".
    moran1/moran2  — spatial coherence of the PC1/PC2 score maps.
    """
    blank = dict(evr1=np.nan, evr2=np.nan, evr3=np.nan, gap23=np.nan,
                 lam2_over_noise=np.nan, moran1=np.nan, moran2=np.nan,
                 eigenvalues=np.zeros(10))
    if spectra.shape[0] < 15:
        return blank
    vectors, eigenvalues, mean_spectrum = bd.run_pca(spectra)
    lam = np.clip(eigenvalues, 0, None)
    total = lam.sum()
    if total <= 0:
        return blank
    ratios = lam / total
    noise = float(lam[NOISE_COMPONENTS].mean())

    out = dict(evr1=float(ratios[0]), evr2=float(ratios[:2].sum()),
               evr3=float(ratios[:3].sum()),
               gap23=float(lam[1] / lam[2]) if lam[2] > 0 else np.inf,
               lam2_over_noise=float(lam[1] / noise) if noise > 0 else np.inf,
               moran1=np.nan, moran2=np.nan, eigenvalues=lam[:10])

    if index is not None:
        scores = bd.compute_pc_scores(spectra, vectors, mean_spectrum, 2)
        out["moran1"] = morans_i(scores[:, 0], yi[index], xi[index], shape)
        out["moran2"] = morans_i(scores[:, 1], yi[index], xi[index], shape)
    return out


def subtract_medium(spectra, medium_abundance, medium, k):
    """r_i = s_i - k * a_i,medium * m  (linear-mixing removal of the medium)."""
    return spectra - k * medium_abundance[:, None] * medium[None, :]


def renormalise(residual, medium_abundance, k):
    """Rescale each residual to unit area, counting negative bins as area.

    The surviving area is measured as sum(|r|), not predicted as 1 - k*a.
    Those agree exactly while the residual stays non-negative, but they part
    company the moment a subtraction overshoots, and they part company in the
    dangerous direction: 1 - k*a is the SIGNED sum, so a deep negative trough
    cancels positive signal, shrinks the denominator, and blows up precisely
    the pixel that was over-subtracted hardest. Counting a trough as area
    treats it as signal that is present but mis-signed, so a worse
    subtraction now yields less amplification rather than more.

    Pixels left with less than RENORM_FLOOR of their original area are almost
    entirely medium; renormalising them would only amplify noise, so they are
    dropped. `medium_abundance` and `k` are kept in the signature for callers
    but are no longer needed to compute the area.
    """
    area = np.abs(residual).sum(axis=1)
    keep = area > RENORM_FLOOR
    out = residual[keep] / area[keep][:, None]
    return out, keep


def sweep(spectra, abundances, medium, cell_flat, yi, xi, shape, rng):
    """Score every k, with two nulls.

    NULL-shuffle : correct medium spectrum, abundances scrambled across space.
                   Removes nothing real but still injects a rank-1 direction —
                   so it exposes any metric that merely rewards rank-1 structure.
    NULL-uniform : the same *average* amount removed from every pixel. With
                   sum-normalised spectra this is an affine map of the data, so
                   PCA eigenvalue ratios are mathematically unchanged; it is
                   here as a sanity check that the code reproduces that.
    """
    a_med = abundances[:, 0]
    cell_idx = np.flatnonzero(cell_flat)
    mean_a = float(a_med[cell_flat].mean())
    rows = []
    for k in K_SWEEP:
        resid = subtract_medium(spectra, a_med, medium, k)

        # (1) cell pixels, renormalised — the primary reading.
        r_cell, keep = renormalise(resid[cell_flat], a_med[cell_flat], k)
        cell_renorm = pca_scores(r_cell, cell_idx[keep], yi, xi, shape)

        # (2) cell pixels, raw residual (magnitude retained).
        cell_raw = pca_scores(resid[cell_flat], cell_idx, yi, xi, shape)

        # (3) every pixel, raw residual.
        all_raw = pca_scores(resid)

        # (4) null A: same k, same abundance distribution, wrong pixels.
        null_evr2, null_l2n, null_moran2 = [], [], []
        for _ in range(N_NULL):
            shuffled = rng.permutation(a_med)
            r_null = subtract_medium(spectra, shuffled, medium, k)
            rn, nkeep = renormalise(r_null[cell_flat], shuffled[cell_flat], k)
            sc = pca_scores(rn, cell_idx[nkeep], yi, xi, shape)
            null_evr2.append(sc["evr2"])
            null_l2n.append(sc["lam2_over_noise"])
            null_moran2.append(sc["moran2"])

        # (5) null B: uniform removal of the mean in-cell medium fraction.
        r_uni = spectra[cell_flat] - k * mean_a * medium[None, :]
        uni = pca_scores(r_uni / (1.0 - k * mean_a), cell_idx, yi, xi, shape)

        rows.append(dict(
            k=float(k),
            cell_renorm_evr1=cell_renorm["evr1"],
            cell_renorm_evr2=cell_renorm["evr2"],
            cell_renorm_gap23=cell_renorm["gap23"],
            cell_renorm_lam2noise=cell_renorm["lam2_over_noise"],
            cell_renorm_moran1=cell_renorm["moran1"],
            cell_renorm_moran2=cell_renorm["moran2"],
            cell_raw_evr2=cell_raw["evr2"],
            all_raw_evr2=all_raw["evr2"],
            null_evr2=float(np.nanmean(null_evr2)),
            null_evr2_sd=float(np.nanstd(null_evr2)),
            null_lam2noise=float(np.nanmean(null_l2n)),
            null_moran2=float(np.nanmean(null_moran2)),
            uniform_evr2=uni["evr2"],
            uniform_lam2noise=uni["lam2_over_noise"],
            n_kept=int(keep.sum()),
            n_dropped=int((~keep).sum()),
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Endmember labelling
# ─────────────────────────────────────────────

def peak_position(spectrum, shift):
    """Sub-bin peak location by parabolic interpolation around the maximum."""
    i = int(np.argmax(spectrum))
    if 0 < i < len(spectrum) - 1:
        y0, y1, y2 = spectrum[i - 1], spectrum[i], spectrum[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            i = i + 0.5 * (y0 - y2) / denom
    return float(np.interp(i, np.arange(len(shift)), shift))


def label_endmembers(endmembers, abundances, shift, cell_mask, yi, xi):
    """Order the two free endmembers as (cytoplasm, nucleus).

    Primary criterion is peak shift — a denser, stiffer nucleus sits at higher
    frequency. Centrality (mean distance from the cell centroid) is computed
    as an independent cross-check, not used to decide, so a disagreement is
    visible rather than silently resolved.
    """
    peaks = [peak_position(endmembers[j], shift) for j in (1, 2)]
    nucleus_local = int(np.argmax(peaks))          # 0 -> EM2, 1 -> EM3

    ys, xs = np.where(cell_mask)
    cy, cx = ys.mean(), xs.mean()
    radius = np.hypot(yi - cy, xi - cx)
    in_cell = cell_mask[yi, xi]
    r_norm = radius[in_cell] / max(radius[in_cell].max(), 1e-9)
    centrality = [
        float(np.average(r_norm, weights=np.clip(abundances[in_cell, j], 1e-9, None)))
        for j in (1, 2)
    ]
    nucleus_by_centrality = int(np.argmin(centrality))

    # nucleus_local indexes `peaks`, i.e. 0 -> endmember row 1, 1 -> row 2.
    nucleus_idx = nucleus_local + 1
    cyto_idx = 2 if nucleus_idx == 1 else 1
    return dict(cyto_idx=cyto_idx, nucleus_idx=nucleus_idx,
                peaks=peaks, centrality=centrality,
                agrees=bool(nucleus_local == nucleus_by_centrality))


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def to_map(values, yi, xi, shape):
    m = np.full(shape, np.nan)
    m[yi, xi] = values
    return m


def render(stem, data, bg_mask, cell_mask, medium, endmembers, abundances,
           labels, sweep_df, k_evr, k_honest, out_path):
    shape = data["shape"]
    yi, xi, shift = data["yi"], data["xi"], data["shift_freq"]
    cyto, nuc = labels["cyto_idx"], labels["nucleus_idx"]
    a_med = abundances[:, 0]
    cell_flat = cell_mask[yi, xi]
    cell_idx = np.flatnonzero(cell_flat)

    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    # ---- Row 1: pure components, RGB composite, per-component fractions
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(shift, endmembers[0], lw=2.2, color="tab:blue",
            label=f"medium (B)  {peak_position(endmembers[0], shift):.3f} GHz")
    ax.plot(shift, endmembers[cyto], lw=1.6, color="tab:green",
            label=f"cytoplasm (G)  {labels['peaks'][cyto-1]:.3f} GHz")
    ax.plot(shift, endmembers[nuc], lw=1.6, color="tab:red",
            label=f"nucleus (R)  {labels['peaks'][nuc-1]:.3f} GHz")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title("three pure endmembers"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    rgb = np.zeros(shape + (3,))
    rgb[yi, xi, 0] = abundances[:, nuc]
    rgb[yi, xi, 1] = abundances[:, cyto]
    rgb[yi, xi, 2] = abundances[:, 0]
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.clip(rgb, 0, 1), origin="lower")
    ax.contour(cell_mask.astype(float), levels=[0.5], colors="w", linewidths=0.8)
    agree = "agree" if labels["agrees"] else "DISAGREE"
    ax.set_title("RGB composite   R=nucleus G=cytoplasm B=medium\n"
                 f"(peak-shift vs centrality labelling: {agree})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    for col, (j, name, cmap) in enumerate(
            [(0, "medium", "Blues"), (nuc, "nucleus", "Reds")], start=2):
        axj = fig.add_subplot(gs[0, col])
        im = axj.imshow(to_map(abundances[:, j], yi, xi, shape), origin="lower",
                        cmap=cmap, vmin=0, vmax=1)
        axj.set_title(f"{name} fraction")
        fig.colorbar(im, ax=axj, shrink=0.8)
        axj.set_xticks([]); axj.set_yticks([])

    # ---- Row 2: cytoplasm fraction, regions, requested metric, honest metric
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(to_map(abundances[:, cyto], yi, xi, shape), origin="lower",
                   cmap="Greens", vmin=0, vmax=1)
    ax.set_title("cytoplasm fraction")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(bg_mask.astype(float) + 2 * cell_mask.astype(float),
              origin="lower", cmap="viridis")
    ax.set_title(f"regions: background {int(bg_mask.sum())} px, "
                 f"cell {int(cell_mask.sum())} px", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(sweep_df["k"], sweep_df["cell_renorm_evr2"], "o-", lw=2,
            color="tab:red", label="real subtraction")
    ax.fill_between(sweep_df["k"],
                    sweep_df["null_evr2"] - sweep_df["null_evr2_sd"],
                    sweep_df["null_evr2"] + sweep_df["null_evr2_sd"],
                    color="0.7", alpha=0.5)
    ax.plot(sweep_df["k"], sweep_df["null_evr2"], "--", lw=1.5, color="0.35",
            label=f"NULL shuffled abundances (n={N_NULL})")
    ax.plot(sweep_df["k"], sweep_df["uniform_evr2"], ":", lw=1.5,
            color="tab:brown", label="NULL uniform removal")
    ax.set_xlabel("k  (fraction of estimated medium removed)")
    ax.set_ylabel("explained variance, PC1+PC2")
    ax.set_title("the requested metric\n"
                 "null above real = metric rewards injected rank-1 structure",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(sweep_df["k"], sweep_df["cell_renorm_lam2noise"], "o-", lw=2,
            color="tab:purple", label="real subtraction")
    ax.plot(sweep_df["k"], sweep_df["null_lam2noise"], "--", lw=1.5,
            color="0.35", label="NULL shuffled")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.axvline(k_honest, color="k", ls=":", lw=1.2)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\lambda_2$ / mean($\lambda_{4..10}$)")
    ax.set_title("scale-free: is PC2 above the noise floor?\n"
                 f"peaks at k = {k_honest:.2f}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- Row 3: spatial coherence, scree, PC2 score maps before/after
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(sweep_df["k"], sweep_df["cell_renorm_moran2"], "o-", lw=2,
            color="tab:green", label="PC2, real")
    ax.plot(sweep_df["k"], sweep_df["null_moran2"], "--", lw=1.5, color="0.35",
            label="PC2, NULL shuffled")
    ax.plot(sweep_df["k"], sweep_df["cell_renorm_moran1"], "-", lw=1.2,
            color="tab:olive", alpha=0.7, label="PC1, real")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("Moran's I")
    ax.set_title("spatial coherence of PC scores\n"
                 "(real structure vs salt-and-pepper)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for k, colour in [(0.0, "0.4"), (1.0, "tab:red")]:
        resid = subtract_medium(data["folded_norm"], a_med, medium, k)
        r_cell, keep = renormalise(resid[cell_flat], a_med[cell_flat], k)
        sc = pca_scores(r_cell, cell_idx[keep], yi, xi, shape)
        lam = np.clip(sc["eigenvalues"], 1e-24, None)
        ax.semilogy(np.arange(1, len(lam) + 1), lam, "o-", lw=1.5, color=colour,
                    label=f"k={k:.0f}  PC1+PC2={sc['evr2']:.3f}")
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue (absolute)")
    ax.set_title("scree - note the flat noise continuum", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for col, k in enumerate([0.0, 1.0], start=2):
        resid = subtract_medium(data["folded_norm"], a_med, medium, k)
        r_cell, keep = renormalise(resid[cell_flat], a_med[cell_flat], k)
        vectors, _, mean_spec = bd.run_pca(r_cell)
        sc2 = bd.compute_pc_scores(r_cell, vectors, mean_spec, 2)[:, 1]
        idx = cell_idx[keep]
        m = np.full(shape, np.nan)
        m[yi[idx], xi[idx]] = sc2
        axj = fig.add_subplot(gs[2, col])
        lim = np.nanpercentile(np.abs(sc2), 98)
        im = axj.imshow(m, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        mi = morans_i(sc2, yi[idx], xi[idx], shape)
        axj.set_title(f"PC2 score map, k={k:.0f}   Moran I={mi:.2f}", fontsize=9)
        fig.colorbar(im, ax=axj, shrink=0.8)
        axj.set_xticks([]); axj.set_yticks([])

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(shift, data["folded_norm"][cell_flat].mean(axis=0), lw=1.6,
            color="0.4", label="mean in-cell, k=0")
    resid = subtract_medium(data["folded_norm"], a_med, medium, 1.0)
    r_cell, keep = renormalise(resid[cell_flat], a_med[cell_flat], 1.0)
    if r_cell.shape[0]:
        ax.plot(shift, r_cell.mean(axis=0), lw=2.0, color="tab:red",
                label="mean in-cell, k=1")
    ax.plot(shift, medium, lw=1.2, color="tab:blue", ls="--", label="medium")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title("in-cell spectrum before / after full removal\n"
                 f"({int((~keep).sum())} px dropped as over-subtracted)",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"{stem} - ternary unmixing, medium removal, PCA scoring",
                 fontsize=12)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)

# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def process(path, rng):
    stem = path.stem
    data = bge.load_cube(path)

    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    # confident_background_mask needs both maps: a pixel must read as
    # background in the raw AND the smoothed BFS to qualify.
    bg_mask, cell_mask, uncertain, info = bge.find_background(
        data, bfs_raw, bfs_sm)
    if int(bg_mask.sum()) < bge.MIN_BG_PIXELS:
        print("  [skip] not enough confident background")
        return None

    cell_flat = cell_mask[data["yi"], data["xi"]]
    n_cell = int(cell_flat.sum())
    if n_cell < 30:
        print(f"  [skip] only {n_cell} cell pixels")
        return None

    medium, _, _ = bge.build_endmember(data, bg_mask)

    spectra = data["folded_norm"]
    endmembers, _ = bge.vca_anchored(spectra, 3, medium, seed=SEED,
                                     candidates=np.flatnonzero(cell_flat))
    abundances = bd.estimate_abundances(spectra, endmembers)

    labels = label_endmembers(endmembers, abundances, data["shift_freq"],
                              cell_mask, data["yi"], data["xi"])

    sweep_df = sweep(spectra, abundances, medium, cell_flat,
                     data["yi"], data["xi"], data["shape"], rng)
    sweep_df.insert(0, "sample", stem)

    # Two selections, kept separate on purpose: the requested criterion and
    # the scale-free one. Where they disagree, the metric is the problem.
    k_evr = float(sweep_df.loc[sweep_df["cell_renorm_evr2"].idxmax(), "k"])
    k_honest = float(sweep_df.loc[sweep_df["cell_renorm_lam2noise"].idxmax(), "k"])

    base = sweep_df.loc[sweep_df["k"] == 0.0].iloc[0]
    full = sweep_df.loc[sweep_df["k"] == 1.0].iloc[0]
    print(f"  cell {n_cell} px | EVR2 {base['cell_renorm_evr2']:.3f} (k=0) "
          f"-> {full['cell_renorm_evr2']:.3f} (k=1); null at k=1 "
          f"{full['null_evr2']:.3f} | lam2/noise "
          f"{base['cell_renorm_lam2noise']:.2f} -> "
          f"{full['cell_renorm_lam2noise']:.2f} | MoranI(PC2) "
          f"{base['cell_renorm_moran2']:.2f} -> "
          f"{full['cell_renorm_moran2']:.2f} | argmax k: EVR2={k_evr:.2f} "
          f"lam2noise={k_honest:.2f} | nucleus label "
          f"{'ok' if labels['agrees'] else 'DISPUTED'}")

    render(stem, data, bg_mask, cell_mask, medium, endmembers, abundances,
           labels, sweep_df, k_evr, k_honest,
           OUTPUT_DIR / f"{stem}_medium_subtraction.png")

    nuc, cyto = labels["nucleus_idx"], labels["cyto_idx"]
    np.savez_compressed(
        OUTPUT_DIR / f"{stem}_unmixing.npz",
        shift=data["shift_freq"], medium=medium, endmembers=endmembers,
        abundances=abundances, yi=data["yi"], xi=data["xi"],
        shape=np.array(data["shape"]), cell_mask=cell_mask, bg_mask=bg_mask,
        nucleus_idx=nuc, cyto_idx=cyto, k_evr=k_evr, k_honest=k_honest)

    return sweep_df, dict(
        sample=stem, n_cell=n_cell, n_background=int(bg_mask.sum()),
        k_argmax_evr2=k_evr, k_argmax_lam2noise=k_honest,
        evr2_k0=float(base["cell_renorm_evr2"]),
        evr2_k1=float(full["cell_renorm_evr2"]),
        null_evr2_k1=float(full["null_evr2"]),
        evr2_beats_null_k1=bool(full["cell_renorm_evr2"] > full["null_evr2"]),
        uniform_evr2_k1=float(full["uniform_evr2"]),
        lam2noise_k0=float(base["cell_renorm_lam2noise"]),
        lam2noise_k1=float(full["cell_renorm_lam2noise"]),
        moran2_k0=float(base["cell_renorm_moran2"]),
        moran2_k1=float(full["cell_renorm_moran2"]),
        moran2_null_k1=float(full["null_moran2"]),
        medium_peak_ghz=peak_position(medium, data["shift_freq"]),
        cyto_peak_ghz=labels["peaks"][cyto - 1],
        nucleus_peak_ghz=labels["peaks"][nuc - 1],
        label_agrees=labels["agrees"],
        mean_nucleus_frac_in_cell=float(abundances[cell_flat, nuc].mean()),
        mean_cyto_frac_in_cell=float(abundances[cell_flat, cyto].mean()),
        mean_medium_frac_in_cell=float(abundances[cell_flat, 0].mean()),
        max_medium_frac_in_cell=float(abundances[cell_flat, 0].max()),
        n_dropped_at_k1=int(full["n_dropped"]),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sweeps, summaries = [], []
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            result = process(path, rng)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        if result:
            sweeps.append(result[0])
            summaries.append(result[1])

    if not summaries:
        return
    pd.concat(sweeps).to_csv(OUTPUT_DIR / "k_sweep_all_samples.csv",
                             index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "medium_subtraction_summary.csv", index=False)

    # Cross-sample view: the requested metric, its null, and the two
    # scale-free readings.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for df in sweeps:
        name = df["sample"].iloc[0][:26]
        axes[0, 0].plot(df["k"], df["cell_renorm_evr2"], lw=1.2, alpha=0.85,
                        label=name)
        axes[0, 1].plot(df["k"], df["cell_renorm_evr2"] - df["null_evr2"],
                        lw=1.2, alpha=0.85)
        axes[1, 0].plot(df["k"], df["cell_renorm_lam2noise"], lw=1.2,
                        alpha=0.85)
        axes[1, 1].plot(df["k"], df["cell_renorm_moran2"], lw=1.2, alpha=0.85)

    axes[0, 0].set_ylabel("explained variance PC1+PC2")
    axes[0, 0].set_title("requested metric — all samples")
    axes[0, 0].legend(fontsize=6, ncol=2)

    axes[0, 1].axhline(0, color="k", lw=1.0)
    axes[0, 1].set_ylabel("real − null")
    axes[0, 1].set_title("margin over shuffled-abundance null\n"
                         "(below 0 = the null wins; metric is not measuring "
                         "medium removal)")

    axes[1, 0].axhline(1.0, color="k", lw=1.0, ls="--")
    axes[1, 0].set_ylabel(r"$\lambda_2$ / mean($\lambda_{4..10}$)")
    axes[1, 0].set_title("scale-free: PC2 above the noise floor")

    axes[1, 1].axhline(0.0, color="k", lw=1.0)
    axes[1, 1].set_ylabel("Moran's I of PC2 scores")
    axes[1, 1].set_title("spatial coherence of the surviving component")

    for ax in axes.ravel():
        ax.set_xlabel("k  (fraction of estimated medium removed)")
        ax.grid(alpha=0.3)
    fig.suptitle("Medium removal: does it help? (in-cell pixels, renormalised)",
                 fontsize=13)
    fig.savefig(OUTPUT_DIR / "k_sweep_all_samples.png", dpi=140)
    plt.close(fig)

    cols = ["sample", "n_cell", "k_argmax_evr2", "evr2_k0", "evr2_k1",
            "null_evr2_k1", "evr2_beats_null_k1", "lam2noise_k0",
            "lam2noise_k1", "moran2_k0", "moran2_k1", "moran2_null_k1",
            "label_agrees"]
    print("\n" + summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
