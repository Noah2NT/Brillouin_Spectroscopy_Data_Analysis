"""How many components does each scan actually support, above its own noise?

Not "how much variance does PC_k explain" -- that is a ratio, and a ratio says
nothing about whether PC_k is real. This asks the absolute question: is the
k-th direction bigger than what pure noise would have produced anyway?

THE METHOD, in three steps.

1. WHITEN. Subtract the known background endmember, D = S - EM_1, then divide
   every element by its own noise sigma:

       sigma_pi = sqrt(alpha * S_pi / C_p)          Z = D / sigma

   This is the step that makes the rest possible. Raw Brillouin spectra are
   wildly heteroscedastic -- sigma varies about 12x between the peak bin and
   the tail, because shot noise follows the signal. An SVD of the raw D is
   therefore dominated by the peak bins for a purely instrumental reason, and
   its singular values cannot be compared against any single noise level.
   After whitening every element carries unit-variance noise by construction,
   so ONE threshold applies to the whole matrix.

2. THRESHOLD. For an N x B matrix of unit-variance noise the largest singular
   value is neither zero nor small -- it concentrates near

       sqrt(N) + sqrt(B)                            (Marchenko-Pastur)

   This is the part that usually gets skipped. With no signal whatever, the
   top singular value of a 450x112 noise matrix still lands near 32. Anything
   under that line is indistinguishable from noise however large it looks in
   isolation. The asymptotic value is checked here against a direct simulation
   of Gaussian matrices of the same shape, so the line is measured rather than
   assumed.

3. COUNT. The number of singular values of Z above that line is the number of
   components the data support BEYOND EM_1. One means two materials in total,
   two means three, zero means the scan holds nothing past the background
   that noise cannot account for.

WHICH PIXELS. Confident-background pixels are excluded. A component living
only in the blob has a singular value that grows with the number of pixels
carrying it, while the threshold grows as sqrt(N) over ALL pixels -- so
padding the matrix with hundreds of pure-background rows raises the bar
without raising the signal, costing sensitivity for nothing.

TWO HONEST CAVEATS.

  * sigma is estimated from the observed spectrum rather than from an unknown
    true mean, so wherever noise dipped low sigma is slightly underestimated
    and Z slightly inflated. Small at these count levels, but it points
    toward over-counting rather than under-counting.
  * The threshold is held fixed across k. Strictly, once k real directions are
    removed the remaining noise matrix is effectively N x (B-k) and the bar
    drops a little. Holding it fixed is the conservative choice.
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


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bge = _load("background_endmember")
tms = _load("two_material_solve")

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "component_count"
N_SIM = 30          # Gaussian matrices drawn to measure the noise threshold
SEED = 0
EPS = 1e-12


def noise_spectrum(n, b, reps=N_SIM, seed=SEED):
    """The WHOLE singular value spectrum of pure noise, index by index.

    A single scalar threshold does not work here, and the reason is worth
    stating because it is easy to get wrong. The largest singular value of a
    noise matrix sits near sqrt(n)+sqrt(b), but the second, third and
    hundredth sit only slightly below it -- the noise spectrum is a dense
    packed band, not a cliff. Comparing every singular value against the
    threshold for the FIRST one therefore lets the entire noise bulk cross in
    one go the moment it drifts marginally high: on known-truth 2-material
    simulation data that rule reported 19 components where the answer is 1.

    So the threshold is a vector: sv_k is judged against the k-th singular
    value of noise, at a high percentile over repeated draws.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([np.linalg.svd(rng.standard_normal((n, b)),
                                    compute_uv=False) for _ in range(reps)])
    return np.percentile(draws, 99, axis=0)


MARGIN = 1.5    # calibrated on known-truth 2-material simulations; see below


def count_components(sv, sim, margin=MARGIN):
    """Leading singular values above this data's own noise band.

    Two corrections on top of the simulated spectrum, both needed, both
    learned from known-truth data rather than assumed:

    SELF-CALIBRATION. Whitening is never exact -- sigma is estimated from
    observed counts rather than true means, so it runs slightly low and every
    singular value comes out uniformly inflated. Measured against an external
    simulation the whole noise bulk then sits 2-20% high and hundreds of
    indices "clear" the bar at once. The trailing half of the spectrum is
    certainly noise, so it is used to rescale the simulated curve onto this
    matrix's own noise level. Any flat scale error cancels.

    MARGIN. Even self-calibrated, the noise band's top edge is fuzzy and a
    real component sitting right on it cannot be called either way. Requiring
    1.5x separates cleanly on the simulation set: there the true component
    runs 1.55x to 31.6x above the band while the first noise direction never
    exceeds 1.21x. That gap is the whole basis for the number, so it is
    stated rather than buried -- on data with a genuinely marginal component
    this rule will miss it, and 1.5 is the knob to lower.

    Stops at the first failure: singular values are ordered, so once one sinks
    into the noise nothing beyond it can be real.
    """
    # Middle-to-late band, not the very end: the last indices of a real
    # spectrum collapse toward zero for rank reasons that the Gaussian
    # simulation does not share, and including them drags the scale factor
    # down until the threshold vanishes and everything "clears" it.
    n = len(sv)
    tail = slice(int(0.55 * n), int(0.90 * n))
    scale = float(np.median(sv[tail] / sim[tail]))
    # GUARD. If whitening worked, the data's own noise band should already sit
    # near the simulated one and `scale` should land near 1. When it does not,
    # the noise model is wrong for this scan and the count that follows would
    # be fiction -- three of the twelve real scans land near 0.18 and would
    # otherwise report ~78 components. Refuse rather than emit a number.
    if not (0.5 <= scale <= 2.0):
        return -1, scale * sim * margin

    thr = scale * sim * margin
    k = 0
    while k < len(sv) and k < len(thr) and sv[k] > thr[k]:
        k += 1
    return k, thr


def whitened(S, EM_1, alpha, counts, sel):
    """Background-subtracted, noise-whitened matrix for the chosen pixels.

    Sigma is floored at ONE COUNT, and that floor is not cosmetic. Poisson
    sigma is sqrt of the MEAN, but all we have is an observation, and an
    observation of zero does not mean sigma is zero. The far edges of the
    folded spectrum hold about 0.02 counts per bin -- they are empty -- so a
    pixel reading exactly 0 there would otherwise get sigma ~ 3e-9 and a
    z-score in the thousands. Ten dead columns like that dominate every
    singular value and make the whole test meaningless.

    Working in counts makes the floor natural: sqrt(alpha * max(c, 1)). Bins
    that carry real signal are untouched (the peak sits near 3000 counts, so
    a floor of 1 changes nothing), while empty bins get the sigma of a single
    count instead of zero.
    """
    Ssel, Csel = S[sel], counts[sel]
    obs_counts = Ssel * Csel[:, None]
    sigma_counts = np.sqrt(alpha * np.maximum(obs_counts, 1.0))
    sigma = sigma_counts / Csel[:, None]
    return (Ssel - EM_1[None, :]) / sigma


def analyse(path):
    data = bge.load_cube(path)
    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])
    bg_mask, _, _, _ = bge.find_background(data, bfs_raw, bfs_sm)
    if int(bg_mask.sum()) < bge.MIN_BG_PIXELS:
        return None

    EM_1, _, _ = bge.build_endmember(data, bg_mask)
    yi, xi = data["yi"], data["xi"]
    bg_flat = bg_mask[yi, xi]
    sel = ~bg_flat                      # blob + uncertain: where signal lives
    if int(sel.sum()) < 40:
        return None

    S = data["folded_norm"]
    counts = data["folded_raw"].sum(axis=1)
    alpha, _, r2 = tms.fit_alpha(data["folded_raw"][bg_flat])

    Z = whitened(S, EM_1, alpha, counts, sel)
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    n, b = Z.shape
    sim = noise_spectrum(n, b)
    n_comp, thr = count_components(sv, sim)

    return dict(sample=path.stem, n_pixels=n, n_bins=b, alpha=alpha,
                ptc_r2=r2, threshold_p95=float(thr[0]),
                threshold_mp=float(np.sqrt(n) + np.sqrt(b)),
                n_components=n_comp,
                sv=sv, thr=thr, Vt=Vt, shift=data["shift_freq"], EM_1=EM_1,
                sv1_ratio=float(sv[0] / thr[0]),
                sv2_ratio=float(sv[1] / thr[1]),
                sv3_ratio=float(sv[2] / thr[2]))


def render(results, out_path):
    fig = plt.figure(figsize=(19, 11), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0:2])
    for r in results:
        k = min(12, len(r["sv"]))
        ax.semilogy(np.arange(1, k + 1), r["sv"][:k] / r["threshold_p95"],
                    "o-", lw=1.2, ms=4, alpha=0.85, label=r["sample"][:26])
    ax.axhline(1.0, color="k", lw=2.0, ls="--", label="noise threshold")
    ax.set_xlabel("component index")
    ax.set_ylabel("singular value / noise threshold")
    ax.set_title("everything below the dashed line is indistinguishable from "
                 "noise", fontsize=11)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2])
    names = [r["sample"][:24] for r in results]
    counts = [r["n_components"] for r in results]
    order = np.argsort(counts)
    ax.barh(range(len(results)), [counts[i] for i in order], color="tab:blue")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("components above noise (beyond EM_1)")
    ax.set_title("how many materials each scan supports", fontsize=11)
    ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[1, 0])
    for r in results:
        ax.scatter(r["sv1_ratio"], r["sv2_ratio"], s=60, color="tab:purple")
        ax.annotate(r["sample"][:16], (r["sv1_ratio"], r["sv2_ratio"]),
                    fontsize=6, xytext=(5, 3), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.axvline(1.0, color="k", lw=1.2, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sv1 / threshold"); ax.set_ylabel("sv2 / threshold")
    ax.set_title("a 3rd material needs sv2 above 1", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    best = max(results, key=lambda r: r["sv1_ratio"])
    ax = fig.add_subplot(gs[1, 1:])
    for j in range(3):
        v = best["Vt"][j]
        if v[np.argmax(np.abs(v))] < 0:
            v = -v                       # sign is arbitrary; orient upward
        ax.plot(best["shift"], v, lw=1.8,
                label=f"direction {j+1}   sv/thr = "
                      f"{best['sv'][j] / best['threshold_p95']:.2f}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("whitened amplitude")
    ax.set_title(f"what the directions look like -- {best['sample'][:34]}\n"
                 "(noise-weighted, so these are not spectra)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("How many components does each scan support above its own "
                 "noise?", fontsize=13)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"[{path.name}]")
        try:
            r = analyse(path)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        if r is None:
            print("  [skip] not enough background or signal pixels")
            continue
        results.append(r)
        print(f"  {r['n_pixels']}x{r['n_bins']} px | alpha {r['alpha']:.3f} | "
              f"threshold {r['threshold_p95']:.1f} (MP {r['threshold_mp']:.1f})"
              f" | sv1..3 = {r['sv'][0]:.1f}, {r['sv'][1]:.1f}, "
              f"{r['sv'][2]:.1f} -> {r['n_components']} component(s) above noise")

    if not results:
        return
    render(results, OUTPUT_DIR / "component_count.png")
    rows = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("sv", "Vt", "shift", "EM_1")}
        for j in range(6):
            row[f"sv{j + 1}"] = float(r["sv"][j])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "component_count.csv", index=False)
    cols = ["sample", "n_pixels", "alpha", "threshold_p95", "sv1", "sv2",
            "sv3", "sv1_ratio", "sv2_ratio", "n_components"]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
