"""Two more pure endmembers, found in the background-subtracted residual.

Continues background_endmember.py / medium_subtraction.py. The steps, in the
order requested:

  1. Isolate confident background, co-add it -> one pure medium spectrum `m`.
  2. HARD-SET every definite-background pixel to `m`. Those pixels are, by
     assertion, 100% medium; keeping their individual noise realisations only
     feeds a noise carpet into the PCA that follows. This is the one step that
     is a *choice* rather than a measurement, so it is reported (how many
     pixels, and how far the average one was moved).
  3. Estimate each pixel's medium potency x_i, then remove it:
         r_i = (s_i - k * x_i * m) / (1 - k * x_i)
     Both s_i and m sum to 1, so the surviving area is exactly 1 - k*x_i and
     the renormalisation is exact, not a fudge. Pixels whose surviving area
     falls under RENORM_FLOOR are dropped rather than amplified -- at k=1 that
     is precisely the hard-set background, which subtracts to zero and leaves
     the residual dataset containing only cell + uncertain pixels.
  4. PCA and VCA(2) on that residual -> the two remaining pure endmembers.

x_i is computed two independent ways, because the obvious choice is circular:

  x_vca  the medium abundance from anchored VCA(3). This is what was asked
         for, but the residual it produces is *by construction* the part of
         each pixel that a 3-endmember model already attributed to EM2 and
         EM3 -- so VCA(2) on it must return roughly EM2 and EM3. That is a
         consistency check, not a discovery.
  x_nn   the largest x with s_i - x*m >= 0 in every bin (a low percentile of
         the per-bin ratio, for noise). Physical, and blind to any endmember
         choice, so the endmembers it yields are genuinely new information.

Both are run. Agreement between the two sets of endmembers is the result;
disagreement means the VCA(3) model was doing the work, not the subtraction.
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
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_decomposition as bd


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bge = _load("background_endmember")     # masks, co-add, anchored VCA
ms = _load("medium_subtraction")        # subtract_medium, renormalise, pca_scores


DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "residual_endmembers"

K_MAIN = 1.0                    # removal strength used for the displayed VCA
K_SWEEP = np.round(np.linspace(0.0, 1.0, 11), 2)
NN_PCT = 2.0                    # percentile of the per-bin ratio for x_nn
NN_FLOOR = 0.01                 # ignore bins where m < this fraction of its max
SEED = 42
MIN_RESIDUAL_PIXELS = 40
AMP_CAP = 2.0                   # max renorm amplification, x the sample median

TRIMAP_CMAP = ListedColormap(["#d9d9d9", "#c0392b", "#2c7fb8"])


# -------------------------------------------------
# The two new pieces
# -------------------------------------------------

def hard_set_background(spectra, bg_flat, medium):
    """Replace every definite-background spectrum with the pure endmember.

    Returns (spectra_copy, mean_L2_move) -- the move is how far the average
    background pixel was displaced, i.e. how much noise this deletes.
    """
    out = spectra.copy()
    moved = np.linalg.norm(spectra[bg_flat] - medium[None, :], axis=1)
    out[bg_flat] = medium
    return out, float(moved.mean()) if moved.size else np.nan


def nonneg_limit(spectra, medium, pct=NN_PCT, floor=NN_FLOOR):
    """Largest x with s - x*m >= 0 in every bin, robustified to a percentile.

    The strict minimum of s/m is set by whichever single bin noise pushed
    lowest, so a low percentile is used instead. Bins where m is negligible
    are excluded: there the ratio is noise/noise and carries no constraint.
    """
    bins = medium > floor * medium.max()
    ratio = spectra[:, bins] / medium[bins][None, :]
    return np.clip(np.percentile(ratio, pct, axis=1), 0.0, 1.0)


def residual_dataset(spectra, x, medium, k):
    """Subtract, renormalise, drop over-subtracted pixels.

    Returns (residual (M, B), index of the surviving pixels in `spectra`).
    """
    resid = ms.subtract_medium(spectra, x, medium, k)
    r, keep = ms.renormalise(resid, x, k)
    return r, np.flatnonzero(keep)


def residual_endmembers(spectra, x, medium, k, vertex_mask=None, seed=SEED):
    """VCA(2) on the background-removed residual -> the two new endmembers.

    Vertices are searched only inside `vertex_mask` (the cell), and only among
    pixels the renormalisation did not blow up. Both guards are necessary:
    renormalising divides by the surviving area 1 - k*x, so a pixel that was
    95% medium comes back multiplied by 20, and VCA -- which by construction
    hunts for the most extreme spectra -- will pick that amplified noise as a
    "pure" endmember every time. The cap is relative to the sample's own
    median amplification so it means the same thing for any x estimator.

    Abundances are still estimated for the whole residual, so the maps show
    how the excluded ring projects onto the endmembers chosen without it.
    """
    r, idx = residual_dataset(spectra, x, medium, k)
    if r.shape[0] < MIN_RESIDUAL_PIXELS:
        return None
    # Amplification must be measured the same way renormalise now measures
    # area, or the cap would guard against a blow-up that no longer happens.
    area = np.abs(ms.subtract_medium(spectra, x, medium, k)).sum(axis=1)
    amp = 1.0 / np.clip(area[idx], 1e-9, None)
    pool = amp <= AMP_CAP * np.median(amp)
    if vertex_mask is not None:
        pool &= vertex_mask[idx]
    if pool.sum() < MIN_RESIDUAL_PIXELS:
        return None
    em, _ = bd.vca(r[pool], num_endmembers=2, seed=seed)
    return em, bd.estimate_abundances(r, em), r, idx, int(pool.sum())


def order_by_peak(em, ab, shift):
    """Sort endmembers low-shift first; carry the abundance columns along."""
    order = np.argsort([ms.peak_position(e, shift) for e in em])
    return em[order], ab[:, order]


def spectral_angle(a, b):
    """Angle between two spectra in degrees -- scale-free similarity."""
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def to_map(values, yi, xi, shape, index=None):
    m = np.full(shape, np.nan)
    if index is None:
        m[yi, xi] = values
    else:
        m[yi[index], xi[index]] = values
    return m


def render(stem, data, bfs_raw, bg_mask, cell_mask, info, medium,
           x_vca, x_nn, em3, res_vca, res_nn, sweep_df, moved, out_path):
    shape, yi, xi = data["shape"], data["yi"], data["xi"]
    shift = data["shift_freq"]
    em_v, ab_v, r_v, idx_v, n_pool = res_vca
    em_n = res_nn[0] if res_nn is not None else None
    cell_flat, bg_flat = cell_mask[yi, xi], bg_mask[yi, xi]

    fig = plt.figure(figsize=(19, 14.5), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    # ---- Row 1: what went in
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma")
    ax.set_title("BFS map (raw fit)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    trimap = np.zeros(shape, dtype=int)
    trimap[cell_mask] = 1
    trimap[bg_mask] = 2
    ax.imshow(trimap, origin="lower", cmap=TRIMAP_CMAP, vmin=0, vmax=2,
              interpolation="nearest")
    ax.set_title(f"STEP 2 - {int(bg_mask.sum())} background px hard-set to m\n"
                 f"(mean move {moved:.2e}) | cell {info['n_cell']} | "
                 f"uncertain {info['n_uncertain']}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(to_map(x_vca, yi, xi, shape), origin="lower", cmap="Blues",
                   vmin=0, vmax=1)
    ax.set_title("medium potency $x_{vca}$\n(anchored VCA(3) abundance)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    other = ~(cell_flat | bg_flat)
    for sel, colour, lab in ((bg_flat, "tab:blue", "background"),
                             (other, "0.6", "uncertain"),
                             (cell_flat, "tab:red", "cell")):
        ax.scatter(x_vca[sel], x_nn[sel], s=4, c=colour, alpha=0.5, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    rho = float(np.corrcoef(x_vca, x_nn)[0, 1])
    ax.set_xlabel("$x_{vca}$"); ax.set_ylabel("$x_{nn}$ (non-negativity limit)")
    ax.set_title(f"do the two potency estimates agree?\nPearson r = {rho:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- Row 2: PCA on the residual
    ax = fig.add_subplot(gs[1, 0])
    for k, colour in ((0.0, "0.4"), (K_MAIN, "tab:red")):
        r, ridx = residual_dataset(data["folded_norm"], x_vca, medium, k)
        r = r[cell_flat[ridx]]          # cell only, so both k use one population
        sc = ms.pca_scores(r)
        lam = np.clip(sc["eigenvalues"], 1e-24, None)
        ax.semilogy(np.arange(1, len(lam) + 1), lam, "o-", lw=1.5, color=colour,
                    label=f"k={k:g}  PC1+PC2={sc['evr2']:.3f}  n={r.shape[0]}")
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue (absolute)")
    ax.set_title("scree, before vs after removal (cell px only)",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(sweep_df["k"], sweep_df["evr2"], "o-", lw=2, color="tab:red")
    ax.set_xlabel("k  (fraction of estimated medium removed)")
    ax.set_ylabel("explained variance PC1+PC2", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(sweep_df["k"], sweep_df["lam2_over_noise"], "s--", lw=1.5,
             color="tab:purple")
    ax2.set_ylabel(r"$\lambda_2$ / mean($\lambda_{4..10}$)", color="tab:purple")
    ax2.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.axvline(K_MAIN, color="k", ls=":", lw=1.0)
    ax.set_title("is a 2nd component still above the noise floor?", fontsize=9)
    ax.grid(alpha=0.3)

    vectors, _, mean_spec = bd.run_pca(r_v)
    scores = bd.compute_pc_scores(r_v, vectors, mean_spec, 2)
    for col in (0, 1):
        axj = fig.add_subplot(gs[1, col + 2])
        lim = np.nanpercentile(np.abs(scores[:, col]), 98)
        imj = axj.imshow(to_map(scores[:, col], yi, xi, shape, idx_v),
                         origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        mi = ms.morans_i(scores[:, col], yi[idx_v], xi[idx_v], shape)
        axj.set_title(f"residual PC{col+1} score   Moran I={mi:.2f}", fontsize=9)
        fig.colorbar(imj, ax=axj, shrink=0.8)
        axj.set_xticks([]); axj.set_yticks([])

    # ---- Row 3: VCA(2) on the residual
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(shift, medium, lw=1.2, color="tab:blue",
            label=f"medium m  ({ms.peak_position(medium, shift):.2f} GHz)")
    for j, colour in enumerate(("tab:green", "tab:red")):
        ax.plot(shift, em_v[j], lw=2.2, color=colour,
                label=f"residual EM{j+1}  "
                      f"({ms.peak_position(em_v[j], shift):.2f} GHz)")
        ax.plot(shift, em3[j + 1], lw=1.2, ls="--", color=colour, alpha=0.8,
                label=f"anchored VCA(3) free EM{j+1}")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title(f"THE TWO NEW ENDMEMBERS (k={K_MAIN:g}, $x_{{vca}}$)\n"
                 f"vertices from {n_pool} un-amplified cell px | dashed = VCA(3)",
                 fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    for j, cmap in enumerate(("Greens", "Reds")):
        axj = fig.add_subplot(gs[2, j + 1])
        imj = axj.imshow(to_map(ab_v[:, j], yi, xi, shape, idx_v),
                         origin="lower", cmap=cmap, vmin=0, vmax=1)
        axj.set_title(f"residual EM{j+1} abundance", fontsize=9)
        fig.colorbar(imj, ax=axj, shrink=0.8)
        axj.set_xticks([]); axj.set_yticks([])

    ax = fig.add_subplot(gs[2, 3])
    if em_n is not None:
        for j, colour in enumerate(("tab:green", "tab:red")):
            ax.plot(shift, em_v[j], lw=2.0, color=colour,
                    label=f"EM{j+1} from $x_{{vca}}$")
            ax.plot(shift, em_n[j], lw=1.4, ls=":", color=colour,
                    label=f"EM{j+1} from $x_{{nn}}$")
        angles = [spectral_angle(em_v[j], em_n[j]) for j in (0, 1)]
        verdict = ("robust" if max(angles) < 5
                   else "SENSITIVE to how x was estimated")
        ax.set_title("same endmembers from the non-circular potency?\n"
                     f"spectral angle {angles[0]:.1f} deg, {angles[1]:.1f} deg"
                     f" - {verdict}", fontsize=9)
        ax.legend(fontsize=6.5)
    ax.set_xlabel("Shift (GHz)"); ax.grid(alpha=0.3)

    fig.suptitle(f"{stem} - background removed, PCA + VCA(2) on the residual",
                 fontsize=12)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


# -------------------------------------------------
# Driver
# -------------------------------------------------

def process(path):
    stem = path.stem
    data = bge.load_cube(path)

    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    bg_mask, cell_mask, _, info = bge.find_background(data, bfs_raw, bfs_sm)
    if int(bg_mask.sum()) < bge.MIN_BG_PIXELS:
        print("  [skip] not enough confident background")
        return None

    medium, _, _ = bge.build_endmember(data, bg_mask)
    yi, xi = data["yi"], data["xi"]
    bg_flat, cell_flat = bg_mask[yi, xi], cell_mask[yi, xi]
    if int(cell_flat.sum()) < 30:
        print(f"  [skip] only {int(cell_flat.sum())} cell pixels")
        return None

    # STEP 2 - the definite background becomes the pure spectrum, exactly.
    spectra, moved = hard_set_background(data["folded_norm"], bg_flat, medium)
    data["folded_norm"] = spectra

    # STEP 3 - two potency estimates.
    em3, _ = bge.vca_anchored(spectra, 3, medium, seed=SEED,
                              candidates=np.flatnonzero(cell_flat))
    x_vca = bd.estimate_abundances(spectra, em3)[:, 0]
    x_nn = nonneg_limit(spectra, medium)

    # STEP 4 - PCA across k, then VCA(2) at K_MAIN.
    # The sweep is scored on CELL PIXELS ONLY, at every k. Scoring the whole
    # residual instead would compare a different population at each k: the
    # background is still present (and amplified up to 20x) at k=0.9 but has
    # subtracted to zero and been dropped by k=1, so the metric would collapse
    # for a reason that has nothing to do with the subtraction. Cell pixels
    # have x < 1, so they are never dropped and the population is fixed.
    rows = []
    for k in K_SWEEP:
        r, idx = residual_dataset(spectra, x_vca, medium, k)
        in_cell = cell_flat[idx]
        r, idx = r[in_cell], idx[in_cell]
        sc = ms.pca_scores(r, idx, yi, xi, data["shape"])
        rows.append(dict(sample=stem, k=float(k), n_kept=int(r.shape[0]),
                         evr1=sc["evr1"], evr2=sc["evr2"],
                         lam2_over_noise=sc["lam2_over_noise"],
                         moran1=sc["moran1"], moran2=sc["moran2"]))
    sweep_df = pd.DataFrame(rows)

    res_vca = residual_endmembers(spectra, x_vca, medium, K_MAIN, cell_flat)
    if res_vca is None:
        print(f"  [skip] residual has fewer than {MIN_RESIDUAL_PIXELS} pixels")
        return None
    em_v, ab_v = order_by_peak(res_vca[0], res_vca[1], data["shift_freq"])
    res_vca = (em_v, ab_v, res_vca[2], res_vca[3], res_vca[4])

    res_nn = residual_endmembers(spectra, x_nn, medium, K_MAIN, cell_flat)
    if res_nn is not None:
        em_n, ab_n = order_by_peak(res_nn[0], res_nn[1], data["shift_freq"])
        res_nn = (em_n, ab_n, res_nn[2], res_nn[3], res_nn[4])

    # Order the VCA(3) free endmembers the same way, so the dashed overlay in
    # the figure compares like with like.
    em3_free, _ = order_by_peak(em3[1:], np.zeros((1, 2)), data["shift_freq"])
    em3 = np.vstack([em3[0], em3_free])

    idx_v = res_vca[3]
    n_kept = len(idx_v)
    frac_bg_dropped = 1.0 - bg_flat[idx_v].sum() / max(bg_flat.sum(), 1)
    angles = ([spectral_angle(em_v[j], res_nn[0][j]) for j in (0, 1)]
              if res_nn is not None else [np.nan, np.nan])
    circ = [spectral_angle(em_v[j], em3[j + 1]) for j in (0, 1)]
    peaks = [ms.peak_position(em_v[j], data["shift_freq"]) for j in (0, 1)]
    sc_k = sweep_df.loc[sweep_df["k"] == K_MAIN].iloc[0]

    print(f"  bg {int(bg_flat.sum())} px hard-set (mean move {moved:.1e}) | "
          f"residual keeps {n_kept}/{spectra.shape[0]} px "
          f"({100 * frac_bg_dropped:.0f}% of background dropped, "
          f"{res_vca[4]} usable as vertices) | "
          f"EVR2 {sweep_df.iloc[0]['evr2']:.3f} -> {sc_k['evr2']:.3f}, "
          f"lam2/noise {sc_k['lam2_over_noise']:.2f} | "
          f"new EMs at {peaks[0]:.2f} / {peaks[1]:.2f} GHz | "
          f"angle vs VCA(3) {circ[0]:.1f}/{circ[1]:.1f} | "
          f"vs x_nn {angles[0]:.1f}/{angles[1]:.1f}")

    render(stem, data, bfs_raw, bg_mask, cell_mask, info, medium, x_vca, x_nn,
           em3, res_vca, res_nn, sweep_df, moved,
           OUTPUT_DIR / f"{stem}_residual_endmembers.png")

    np.savez_compressed(
        OUTPUT_DIR / f"{stem}_residual_endmembers.npz",
        shift=data["shift_freq"], medium=medium, endmembers=em_v,
        abundances=ab_v, residual_index=idx_v, x_vca=x_vca, x_nn=x_nn,
        endmembers_from_xnn=(res_nn[0] if res_nn is not None else np.zeros(0)),
        vca3_endmembers=em3, yi=yi, xi=xi, shape=np.array(data["shape"]),
        cell_mask=cell_mask, bg_mask=bg_mask, k=K_MAIN)

    return sweep_df, dict(
        sample=stem, n_pixels=int(spectra.shape[0]),
        n_background=int(bg_flat.sum()), n_cell=int(cell_flat.sum()),
        bg_mean_move=moved, n_residual=n_kept, n_vertex_pool=res_vca[4],
        frac_background_dropped=float(frac_bg_dropped),
        x_vca_mean_in_cell=float(x_vca[cell_flat].mean()),
        x_nn_mean_in_cell=float(x_nn[cell_flat].mean()),
        x_corr=float(np.corrcoef(x_vca, x_nn)[0, 1]),
        evr2_k0=float(sweep_df.iloc[0]["evr2"]), evr2_kmain=float(sc_k["evr2"]),
        lam2noise_k0=float(sweep_df.iloc[0]["lam2_over_noise"]),
        lam2noise_kmain=float(sc_k["lam2_over_noise"]),
        moran2_kmain=float(sc_k["moran2"]),
        em1_peak_ghz=peaks[0], em2_peak_ghz=peaks[1],
        medium_peak_ghz=ms.peak_position(medium, data["shift_freq"]),
        angle_em1_vs_vca3=circ[0], angle_em2_vs_vca3=circ[1],
        angle_em1_xvca_vs_xnn=angles[0], angle_em2_xvca_vs_xnn=angles[1],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    sweeps, summaries = [], []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if only and not any(o.lower() in path.name.lower() for o in only):
            continue
        print(f"[{path.name}]")
        try:
            result = process(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        if result:
            sweeps.append(result[0])
            summaries.append(result[1])

    if not summaries:
        return
    pd.concat(sweeps).to_csv(OUTPUT_DIR / "residual_k_sweep.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "residual_endmembers_summary.csv", index=False)
    cols = ["sample", "n_cell", "n_residual", "frac_background_dropped",
            "x_corr", "evr2_k0", "evr2_kmain", "lam2noise_kmain",
            "medium_peak_ghz", "em1_peak_ghz", "em2_peak_ghz",
            "angle_em1_vs_vca3", "angle_em1_xvca_vs_xnn"]
    print("\n" + summary[cols].to_string(index=False))


def demo():
    """Self-check on the two new pieces: the potency limit and the renorm."""
    grid = np.linspace(0, 8, 80)

    def peak(centre):
        s = np.exp(-((grid - centre) ** 2) / 0.08)
        return s / s.sum()

    m, cell = peak(3.0), peak(4.5)

    # A pure-medium pixel reads x ~ 1; mixtures read close to their true share
    # and, crucially, monotonically.
    assert nonneg_limit(m[None, :], m)[0] > 0.97
    xs = [nonneg_limit((f * m + (1 - f) * cell)[None, :], m)[0]
          for f in (0.3, 0.5, 0.7)]
    assert np.all(np.diff(xs) > 0), xs
    assert abs(xs[2] - 0.7) < 0.1, xs

    # Exact removal of a known mix returns the pure cell spectrum.
    mix = (0.7 * m + 0.3 * cell)[None, :]
    r, idx = residual_dataset(mix, np.array([0.7]), m, 1.0)
    assert len(idx) == 1 and np.allclose(r[0], cell, atol=1e-9)

    # A 100% medium pixel is dropped at k=1 rather than renormalised to noise.
    _, idx = residual_dataset(m[None, :], np.array([1.0]), m, 1.0)
    assert len(idx) == 0

    # Hard-setting overwrites exactly the flagged pixels and nothing else.
    data = np.vstack([m + 0.001, cell])
    out, moved = hard_set_background(data, np.array([True, False]), m)
    assert np.allclose(out[0], m) and np.allclose(out[1], cell) and moved > 0
    print("demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
