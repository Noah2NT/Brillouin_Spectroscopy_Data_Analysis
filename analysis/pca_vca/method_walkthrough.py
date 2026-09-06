"""Step-by-step visual walkthrough of the background-endmember method.

One 16-panel figure per sample, in the order the algorithm actually runs:

  INPUTS      1 raw BFS map          2 smoothed BFS map
              3 BFS histogram + every cutoff
  GATE 1      4 value test: raw <= cut AND smoothed <= cut
  GATE 2      5 cell mask (Otsu -> fill -> open -> largest component)
              6 suspicious set = failed gate 1, or cell, or invalid
              7 suspicious dilated by `margin` -> the keep-out zone
  GATE 3      8 final trimap after the minimum-component-size filter
  RESULT      9 stage funnel  10 co-add SNR gain
             11 endmember vs mean in-cell   12 their difference
  USE        13 anchored VCA(3) endmembers  14-16 abundance maps

Written for the three scans named on the command line (defaults below), but
any CSV in data/csv works: `python method_walkthrough.py <stem> [<stem> ...]`.
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
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_decomposition as bd
import feature_extraction as fe


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bge = _load("background_endmember")
cmp_mod = _load("cell_mask_pipeline")

OUTPUT_DIR = Path("outputs") / "background_endmember" / "walkthrough"

DEFAULT_SAMPLES = [
    "Sample11_JiPSC-1000_Plate2_A1_FinalRawData.csv",
    "Sample12_JiPSC-1066_Plate2_A2_data1_new.csv",
    "XY_scanning_40um_60um_step1um_10s_fixed.csv",
]

C_BG, C_UNC, C_CELL = "#2c7fb8", "#d9d9d9", "#c0392b"
TRI_CMAP = ListedColormap([C_UNC, C_CELL, C_BG])
BIN_CMAP = ListedColormap(["#f2f2f2", "#2c7fb8"])
SUS_CMAP = ListedColormap(["#f2f2f2", "#e08214"])


def stages(data, bfs_raw, bfs_sm):
    """Reproduce every intermediate mask so each step can be drawn."""
    valid = data["valid"]
    k = bge.BG_K_MAD
    margin, min_size = bge.BG_MARGIN, bge.BG_MIN_SIZE

    _, cell, thr_s = cmp_mod.mask_from_bfs(bfs_sm, valid)
    thr_r = cmp_mod.otsu_threshold(bfs_raw[valid])
    med_r, mad_r = cmp_mod._robust_background_level(bfs_raw[valid], thr_r)
    med_s, mad_s = cmp_mod._robust_background_level(bfs_sm[valid], thr_s)
    cut_r, cut_s = med_r + k * mad_r, med_s + k * mad_s

    finite = valid & np.isfinite(bfs_raw) & np.isfinite(bfs_sm)
    bg_like = finite & (bfs_raw <= cut_r) & (bfs_sm <= cut_s)
    suspicious = (~bg_like) | cell
    suspicious_d = fe.dilate_mask(suspicious, iterations=margin).astype(bool)
    after_margin = bg_like & ~suspicious_d
    background = cmp_mod._keep_components_at_least(after_margin, min_size)
    uncertain = valid & ~(background | cell)

    return dict(cell=cell, bg_like=bg_like, suspicious=suspicious,
                suspicious_d=suspicious_d, after_margin=after_margin,
                background=background, uncertain=uncertain,
                thr_raw=thr_r, thr_sm=thr_s, med_raw=med_r, mad_raw=mad_r,
                cut_raw=cut_r, cut_sm=cut_s, k=k, margin=margin,
                min_size=min_size)


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])


def render(stem, data, bfs_raw, bfs_sm, st, endmember, summed, bg_idx,
           em_anc, ab_anc, out_path):
    shape = data["shape"]
    yi, xi, shift = data["yi"], data["xi"], data["shift_freq"]
    valid = data["valid"]
    bg, cell, unc = st["background"], st["cell"], st["uncertain"]

    fig = plt.figure(figsize=(21, 20), constrained_layout=True)
    gs = GridSpec(4, 4, figure=fig)

    # ── 1 raw BFS ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(bfs_raw, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_title("1 · raw BFS map\nLorentzian fit, every pixel", fontsize=10)
    _bare(ax)

    # ── 2 smoothed BFS ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(bfs_sm, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GHz")
    ax.set_title(f"2 · smoothed BFS (sigma={bge.MASK_SIGMA})\n"
                 "spreads cell signal into neighbours", fontsize=10)
    _bare(ax)

    # ── 3 histogram + cutoffs ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    vals = bfs_raw[valid]
    lo, hi = np.nanpercentile(vals, [0.2, 99.8])
    bins = np.linspace(lo, hi, 70)
    ax.hist([bfs_raw[bg], bfs_raw[unc], bfs_raw[cell]], bins=bins,
            stacked=True, color=[C_BG, C_UNC, C_CELL], edgecolor="none",
            label=["background", "uncertain", "cell"])
    ax.axvline(st["med_raw"], color="navy", ls=":", lw=1.6,
               label=f"bg median {st['med_raw']:.4f}")
    ax.axvline(st["cut_raw"], color="navy", ls="-", lw=2.0,
               label=f"cut = med+{st['k']:g}·MAD {st['cut_raw']:.4f}")
    ax.axvline(st["thr_sm"], color="k", ls="--", lw=1.6,
               label=f"Otsu {st['thr_sm']:.4f}")
    ax.set_xlabel("BFS (GHz)"); ax.set_ylabel("pixels")
    ax.set_title("3 · where the cutoffs fall", fontsize=10)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

    # ── 4 gate 1 ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(st["bg_like"].astype(int), origin="lower", cmap=BIN_CMAP,
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"4 · GATE 1 value test\nraw<=cut AND smoothed<=cut  "
                 f"[{int(st['bg_like'].sum())} px]", fontsize=10)
    _bare(ax)

    # ── 5 cell mask ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cell.astype(int), origin="lower",
              cmap=ListedColormap(["#f2f2f2", C_CELL]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"5 · cell mask\nOtsu → fill → open → largest CC  "
                 f"[{int(cell.sum())} px]", fontsize=10)
    _bare(ax)

    # ── 6 suspicious ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(st["suspicious"].astype(int), origin="lower", cmap=SUS_CMAP,
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"6 · suspicious set\nfailed gate 1, or cell, or invalid  "
                 f"[{int(st['suspicious'].sum())} px]", fontsize=10)
    _bare(ax)

    # ── 7 dilated keep-out ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    grown = st["suspicious_d"] & ~st["suspicious"]
    layer = np.zeros(shape, dtype=int)
    layer[st["suspicious"]] = 1
    layer[grown] = 2
    ax.imshow(layer, origin="lower", vmin=0, vmax=2,
              cmap=ListedColormap(["#f2f2f2", "#e08214", "#8c2d04"]),
              interpolation="nearest")
    ax.set_title(f"7 · GATE 2 dilate by margin={st['margin']} px\n"
                 f"dark = ring added ({int(grown.sum())} px) → "
                 f"{int(st['after_margin'].sum())} survive", fontsize=10)
    _bare(ax)

    # ── 8 final trimap ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    trimap = np.zeros(shape, dtype=int)
    trimap[cell] = 1
    trimap[bg] = 2
    ax.imshow(trimap, origin="lower", cmap=TRI_CMAP, vmin=0, vmax=2,
              interpolation="nearest")
    ax.set_title(f"8 · GATE 3 min component {st['min_size']} px → FINAL\n"
                 f"bg {int(bg.sum())} | unc {int(unc.sum())} | "
                 f"cell {int(cell.sum())}", fontsize=10)
    ax.legend(handles=[Patch(facecolor=C_BG, label="background"),
                       Patch(facecolor=C_UNC, label="uncertain (excluded)"),
                       Patch(facecolor=C_CELL, label="cell")], fontsize=7,
              loc="upper right")
    _bare(ax)

    # ── 9 stage funnel ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    names = ["valid", "gate 1\nvalue", "gate 2\nmargin", "gate 3\nsize"]
    counts = [int(valid.sum()), int(st["bg_like"].sum()),
              int(st["after_margin"].sum()), int(bg.sum())]
    ax.barh(range(4), counts, color=["#bbbbbb", "#7fcdbb", "#41b6c4", C_BG])
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=8)
    ax.set_xlim(0, counts[0] * 1.18)
    ax.set_xlabel("pixels surviving")
    ax.set_title("9 · attrition through the three gates", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # ── 10 co-add SNR ────────────────────────────────────────────
    mid = bg_idx[len(bg_idx) // 2]
    single = data["folded_norm"][mid]
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(shift, single, color="0.65", lw=1.0,
            label=f"one bg pixel (SNR≈{bge.peak_snr(data['folded_raw'][mid]):.0f})")
    ax.plot(shift, endmember, color=C_BG, lw=2.2,
            label=f"co-added n={len(bg_idx)} "
                  f"(SNR≈{bge.peak_snr(summed):.0f})")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    gain = (bge.baseline_noise(single)
            / max(bge.baseline_noise(endmember), 1e-30))
    ax.set_title(f"10 · co-add the background pixels\n"
                 f"baseline noise ↓{gain:.0f}×", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    tail = bge.tail_slice(len(shift))
    axin = ax.inset_axes([0.55, 0.45, 0.43, 0.33])
    axin.plot(shift[tail], single[tail], color="0.65", lw=0.9)
    axin.plot(shift[tail], endmember[tail], color=C_BG, lw=1.4)
    axin.set_title("baseline tail", fontsize=6)
    axin.tick_params(labelsize=5.5); axin.grid(alpha=0.3)

    # ── 11 endmember vs in-cell ──────────────────────────────────
    cell_flat = cell[yi, xi]
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(shift, endmember, color=C_BG, lw=2.0, label="background endmember")
    if cell_flat.any():
        ax.plot(shift, data["folded_norm"][cell_flat].mean(axis=0),
                color=C_CELL, lw=1.8, ls="--", label="mean in-cell")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title("11 · the pure endmember vs the cell", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── 12 difference ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    if cell_flat.any():
        diff = data["folded_norm"][cell_flat].mean(axis=0) - endmember
        ax.plot(shift, diff, color="k", lw=1.8)
        ax.fill_between(shift, 0, diff, where=diff > 0, color=C_CELL,
                        alpha=0.35)
        ax.fill_between(shift, 0, diff, where=diff < 0, color=C_BG, alpha=0.35)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shift (GHz)")
    ax.set_title("12 · in-cell − background\n"
                 "derivative shape = a shifted peak, not an added one",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # ── 13 anchored endmembers ───────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(shift, em_anc[0], lw=2.4, color=C_BG, label="EM1 = background")
    for j in (1, 2):
        ax.plot(shift, em_anc[j], lw=1.4, label=f"EM{j+1} (free vertex)")
    ax.set_xlabel("Shift (GHz)"); ax.set_ylabel("Norm. intensity")
    ax.set_title("13 · anchored VCA(3)\nbackground pinned as vertex 1",
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── 14-16 abundance maps ─────────────────────────────────────
    for j in range(3):
        axj = fig.add_subplot(gs[3, j + 1])
        m = np.full(shape, np.nan)
        m[yi, xi] = ab_anc[:, j]
        im = axj.imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=1)
        tag = "  (background)" if j == 0 else ""
        axj.set_title(f"{14 + j} · EM{j+1} abundance{tag}", fontsize=10)
        fig.colorbar(im, ax=axj, shrink=0.8)
        _bare(axj)

    fig.suptitle(f"{stem}\nbackground isolation → pure endmember → "
                 f"anchored VCA", fontsize=14)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def process(name):
    path = bge.DATA_DIR / name
    stem = path.stem
    data = bge.load_cube(path)

    bfs_raw, _, _ = bge.bfs_from_flat(data["folded_norm"], data["shift_freq"],
                                      data["yi"], data["xi"], data["shape"])
    flat_sm = bge.smooth_flat(data["folded_norm"], data["yi"], data["xi"],
                              data["shape"], sigma=bge.MASK_SIGMA)
    bfs_sm, _, _ = bge.bfs_from_flat(flat_sm, data["shift_freq"],
                                     data["yi"], data["xi"], data["shape"])

    st = stages(data, bfs_raw, bfs_sm)
    bg = st["background"]
    endmember, summed, bg_idx = bge.build_endmember(data, bg)

    all_flat = data["folded_norm"]
    em_anc, _ = bge.vca_anchored(all_flat, 3, endmember, seed=42)
    ab_anc = bd.estimate_abundances(all_flat, em_anc)

    render(stem, data, bfs_raw, bfs_sm, st, endmember, summed, bg_idx,
           em_anc, ab_anc, OUTPUT_DIR / f"{stem}_walkthrough.png")

    cell_flat = st["cell"][data["yi"], data["xi"]]
    bg_flat = bg[data["yi"], data["xi"]]
    mid = bg_idx[len(bg_idx) // 2]
    row = dict(
        sample=stem, n_valid=int(data["valid"].sum()),
        n_gate1=int(st["bg_like"].sum()),
        n_gate2=int(st["after_margin"].sum()),
        n_background=int(bg.sum()), n_cell=int(st["cell"].sum()),
        n_uncertain=int(st["uncertain"].sum()),
        cut_raw=st["cut_raw"], otsu_smoothed=st["thr_sm"],
        snr_single=bge.peak_snr(data["folded_raw"][mid]),
        snr_coadded=bge.peak_snr(summed),
        noise_gain=float(bge.baseline_noise(data["folded_norm"][mid])
                         / max(bge.baseline_noise(endmember), 1e-30)),
        em1_abundance_in_bg=float(np.mean(ab_anc[bg_flat, 0])),
        em1_abundance_in_cell=float(np.mean(ab_anc[cell_flat, 0])),
    )
    print(f"  valid {row['n_valid']} -> gate1 {row['n_gate1']} -> "
          f"gate2 {row['n_gate2']} -> bg {row['n_background']} | "
          f"SNR {row['snr_single']:.0f}->{row['snr_coadded']:.0f} | "
          f"EM1 in bg {row['em1_abundance_in_bg']:.3f} / "
          f"in cell {row['em1_abundance_in_cell']:.3f}")
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or DEFAULT_SAMPLES
    resolved = []
    for n in names:
        n = n if n.endswith(".csv") else f"{n}.csv"
        if not (bge.DATA_DIR / n).exists():
            hits = sorted(bge.DATA_DIR.glob(f"{Path(n).stem.split('_')[0]}*"))
            if not hits:
                print(f"[skip] {n}: not found in {bge.DATA_DIR}")
                continue
            n = hits[0].name
        resolved.append(n)

    rows = []
    for n in resolved:
        print(f"[{n}]")
        rows.append(process(n))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "walkthrough_summary.csv", index=False)
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
