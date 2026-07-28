"""
Drug dose-series line-scan visualisations (carboplatin / paclitaxel).

Reads every scan under '<BASE>/<folder>/' and produces five figures for the
chosen signal:

  1. Per-folder grid: one panel per scan, region shading + edges from the
     shift-based tabletop fit, title = filename, no legend.
  2. Per-folder overlay: every scan in the folder on a single axes, one
     colour per folder, centre-aligned on its own fitted cell midpoint.
  3. All folders overlaid on ONE axes, colour-coded.
  4. Average line chart: one connected mean line per folder, the point-wise
     mean across samples on a common centre-aligned grid.
  5. Table tops: one panel per folder (4 for carboplatin, 5 for paclitaxel)
     showing the pooled points, the refit tabletop, and the media level in
     grey.
  6. Average panels: bfs_0 and lw_0 averaged side by side in one figure,
     drawn every run regardless of SIGNAL_KIND.

Scans are combined by shifting each one so the centre of its fitted tabletop
sits at x = 0, then pooling every point into a single dataset and refitting
one tabletop. Absolute stage position is discarded; cells of differing size
are simply centred on each other.

The plotted quantity is chosen by SIGNAL_KIND at the bottom of the file:
  'bfs'    - Lorentzian-fit Brillouin frequency shift (GHz)
  'lw'     - Lorentzian-fit Brillouin linewidth (GHz)
  'bfs_0'  - BFS minus that scan's tabletop baseline (BFS - BFS0)
  'lw_0'   - linewidth minus its tabletop baseline (LW - LW0)
  'simple' - pixel spectral gradient with the [-1, 1] kernel across pixels
  'sobel'  - Sobel-Feldman kernel across pixels
  'lap'    - Laplacian kernel across pixels
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_fitting as bf


# ─────────────────────────────────────────────
# Dataset config - uncomment ONE block
# ─────────────────────────────────────────────

# Stage step between adjacent pixels. The X (csv) and Z (xlsx) columns hold
# step INDICES, not microns, so positions are multiplied by this to get um.
STEP_UM = 5.0

# --- PACLITAXEL ---------------------------------------------------------
'''
DRUG    = "Paclitaxel"
BASE    = "Paclitaxel final line scans"
FOLDERS = ["Control", "1.52 nM", "3.125 nM", "6.25 nM", "12.5 nM"]
FOLDER_COLORS = {
     "Control":  "green",
     "1.52 nM":  "blue",
     "3.125 nM": "orange",
     "6.25 nM":  "red",
     "12.5 nM":  "purple",
}
FOLDER_LABELS = {}      # folder names are already display-ready
'''
# --- CARBOPLATIN --------------------------------------------------------
DRUG    = "Carboplatin"
BASE    = "1.5.26 20x Carboplatin"
FOLDERS = ["control", "100um", "200um", "400um"]
FOLDER_COLORS = {
    "control": "green",
    "100um":   "blue",
    "200um":   "orange",
    "400um":   "red",
}
# Folder names are directory names; these are what appears on the charts.
# uM here is micromolar (concentration) - not to be confused with the um in
# the position axis, which is micrometres.
FOLDER_LABELS = {
    "control": "Control",
    "100um":   "100 µM",
    "200um":   "200 µM",
    "400um":   "400 µM",
}


# Write every figure to figures/<DRUG>/ as well as showing it.
SAVE_FIGS = False

REGION_COLORS = {"outside": "#dddddd", "edge": "#ffd9b3", "center": "#b3e6ff"}

SIGNAL_LABELS = {
    "bfs":    "Brillouin shift (GHz)",
    "lw":     "Brillouin linewidth (GHz)",
    "bfs_0":  "BFS - BFS₀ (MHz)",
    "lw_0":   "LW - LW₀ (MHz)",
    "simple": "|spectral gradient| - kernel [-1, 1]",
    "sobel":  "|spectral gradient| - Sobel [1, 2, 0, -2, -1]",
    "lap":    "|spectral gradient| - Laplacian [1, -2, 1]",
}

# The baseline-corrected signals are only a few hundred MHz, so they are
# stored and plotted in MHz rather than GHz (1 GHz = 1000 MHz). Converting at
# the source means the axis, the fitted height and the variance are all in the
# same unit with no per-plot rescaling. Raw bfs/lw stay in GHz - 5.69 GHz
# would read as an unwieldy 5690 MHz.
MHZ_PER_GHZ = 1000.0
MHZ_SIGNALS = {"bfs_0", "lw_0"}

# Signals that have a meaningful tabletop (media -> cell -> media) shape.
TABLETOP_SIGNALS = {"bfs", "lw", "bfs_0", "lw_0"}

# Signals whose plateau may sit BELOW the surrounding medium, so the tabletop
# fit must be allowed a negative amplitude.
NEGATIVE_OK_SIGNALS = {"lw", "lw_0"}

# Averaging (see _aligned_mean). A grid point is only plotted where at least
# this fraction of a folder's scans reach it - otherwise the ends of the mean
# line rest on one scan while the middle rests on all of them.
MEAN_MIN_FRAC = 0.5

# Shade +/- 1 SD across samples around each mean line.
MEAN_SHOW_SD = False


# ─────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "font.weight":      "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth":   1.3,
})


# Axis text for the scan position. This is micrometres (um), a length - the
# uM in the carboplatin dose labels is micromolar, a concentration.
POS_LABEL = "Position relative to cell centre (µm)"


def _flabel(folder):
    """Display name for a dose folder."""
    return FOLDER_LABELS.get(folder, folder)


def _ascii(text):
    """Plain-ASCII version of a label, for printing.

    Figures use µ and subscripts happily, but Windows consoles frequently
    default to cp1252 and raise UnicodeEncodeError on them - which would kill
    a run part-way through simply because its output was redirected to a file.
    """
    return (text.replace("µ", "u")
                .replace("₀", "0")
                .replace("₁", "1"))


def _style_axes(ax):
    """Bold the tick labels - rcParams font.weight does not cover them."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _finish(fig, name):
    """Tidy layout, optionally save, then show."""
    fig.tight_layout()
    if SAVE_FIGS:
        outdir = os.path.join("figures", DRUG)
        os.makedirs(outdir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        fig.savefig(os.path.join(outdir, safe + ".png"),
                    dpi=300, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def _split_indices_by_pass(positions):
    n = len(positions)
    if n == 0:
        return []
    starts = [0]
    direction = 0
    for i in range(1, n):
        step = positions[i] - positions[i - 1]
        if direction == 0:
            if step != 0:
                direction = 1 if step > 0 else -1
            continue
        if direction * step < 0:
            starts.append(i)
            direction = 0
    starts.append(n)
    return [np.arange(starts[k], starts[k + 1])
            for k in range(len(starts) - 1)]


def _sorted_scan(spectra, freq, positions, fname, **extra):
    """Build a scan dict with positions in ascending um.

    Reverse sweeps come back descending; fit_tabletop reads the first and last
    slice of the profile as the outside-the-cell level, so everything
    downstream is simpler if position is always increasing.
    """
    positions = np.asarray(positions, dtype=float) * STEP_UM
    order = np.argsort(positions)
    scan = {
        "spectra":   np.asarray(spectra, dtype=float)[order],
        "freq":      np.asarray(freq, dtype=float),
        "positions": positions[order],
        "filename":  fname,
    }
    for key, val in extra.items():
        scan[key] = (np.asarray(val, dtype=float)[order]
                     if val is not None else None)
    return scan


def _load_csv_scans(full, fname):
    """One CSV may hold several sweeps over the same line; split them."""
    out = []
    df = pd.read_csv(full)
    lengths = df["Raw Data"].apply(lambda s: len(s.split(","))).to_numpy()
    unique_lens = sorted(set(lengths))
    if len(unique_lens) > 1:
        print(f"  [info] {fname}: mixed Raw Data lengths {unique_lens} "
              f"- grouping by length")
    axis = bp._detect_scan_axis(df)
    for L in unique_lens:
        sub = df.loc[lengths == L].reset_index(drop=True)
        positions_all = sub[axis].to_numpy()
        passes = _split_indices_by_pass(positions_all)
        if len(passes) > 1:
            print(f"  [info] {fname} (len={L}): {len(passes)} sweeps "
                  f"detected - splitting into {len(passes)} scans")
        for idx in passes:
            if len(idx) < 10:
                print(f"  [info] {fname} (len={L}): skipping sweep with "
                      f"only {len(idx)} pixel(s)")
                continue
            sub_pass = sub.iloc[idx].reset_index(drop=True)
            spectra = np.vstack(
                sub_pass["Raw Data"].apply(
                    lambda s: np.array([float(v) for v in s.split(",")]))
            )
            start = sub_pass["startFrequency"].iloc[0]
            end   = sub_pass["endFrequency"].iloc[0]
            freq  = np.linspace(start, end, spectra.shape[1])
            out.append(_sorted_scan(spectra, freq,
                                    sub_pass[axis].to_numpy(), fname))
    return out


def _load_xlsx_scan(full, fname):
    """Read a '*_Fitted.xlsx' produced by the acquisition software.

    Sheet 'Raw Data' holds the frequency axis in column 0 and one 512-channel
    spectrum per column after it. Sheet 'Fitted' holds the position index Z
    plus the vendor's own Lorentzian fit, which is kept only as a cross-check
    - the BFS and linewidth used everywhere else are refitted from the raw
    spectra so that xlsx and csv files go through an identical pipeline.
    """
    raw = pd.read_excel(full, sheet_name="Raw Data")
    freq = raw.iloc[:, 0].to_numpy(dtype=float)
    spectra = raw.iloc[:, 1:].to_numpy(dtype=float).T      # (n_pos, n_bins)

    fitted = pd.read_excel(full, sheet_name="Fitted")
    positions = fitted["Z"].to_numpy(dtype=float)
    vendor_bfs = (fitted["BFS_Lorentz"].to_numpy(dtype=float)
                  if "BFS_Lorentz" in fitted.columns else None)
    vendor_lw = (fitted["LW_Lorentz"].to_numpy(dtype=float)
                 if "LW_Lorentz" in fitted.columns else None)

    n = min(len(positions), spectra.shape[0])
    if len(positions) != spectra.shape[0]:
        print(f"  [warn] {fname}: 'Fitted' has {len(positions)} rows but "
              f"'Raw Data' has {spectra.shape[0]} spectra - using first {n}")
    positions = positions[:n] - positions[:n].min()
    return [_sorted_scan(spectra[:n], freq, positions, fname,
                         vendor_bfs=None if vendor_bfs is None else vendor_bfs[:n],
                         vendor_lw=None if vendor_lw is None else vendor_lw[:n])]


def _load_line_scan_folder_robust(folder_path):
    out = []
    for fname in sorted(os.listdir(folder_path)):
        low = fname.lower()
        full = os.path.join(folder_path, fname)
        if low.endswith(".csv"):
            out.extend(_load_csv_scans(full, fname))
        elif low.endswith(".xlsx") and not fname.startswith("~$"):
            out.extend(_load_xlsx_scan(full, fname))
    return out


# ─────────────────────────────────────────────
# Per-scan analysis: shift + spectral-gradient kernels
# ─────────────────────────────────────────────

def spectral_gradient_per_pixel(spectra, kind):
    """Apply a 1D kernel ACROSS pixels; return |summed response| per pixel."""
    n = spectra.shape[0]
    out = np.full(n, np.nan)
    if kind == "simple":       # [-1, 1] forward difference
        for i in range(n - 1):
            out[i] = np.sum(np.abs(spectra[i + 1] - spectra[i]))
    elif kind == "lap":        # [1, -2, 1]
        for i in range(1, n - 1):
            r = spectra[i - 1] - 2.0 * spectra[i] + spectra[i + 1]
            out[i] = np.sum(np.abs(r))
    elif kind == "sobel":      # [1, 2, 0, -2, -1]
        for i in range(2, n - 2):
            r = (spectra[i - 2] + 2.0 * spectra[i - 1]
                 - 2.0 * spectra[i + 1] - spectra[i + 2])
            out[i] = np.sum(np.abs(r))
    else:
        raise ValueError(f"unknown gradient kind: {kind}")
    return out


def _media_level(sig, tab_params, region):
    """The medium ('outside') level of a profile - the bottom of its tabletop.

    Falls back to the mean over pixels the shift tabletop labelled 'outside'
    if the fit for this particular signal failed.
    """
    if tab_params is not None and np.isfinite(tab_params["baseline"]):
        return float(tab_params["baseline"])
    if region is not None:
        m = (region == "outside") & np.isfinite(sig)
        if m.any():
            return float(np.mean(sig[m]))
    return None


def load_and_analyse(folder_path):
    """
    For every scan: fit Brillouin shift and linewidth, compute pixel gradients
    for each kernel kind, and fit a tabletop to both the shift (which drives
    region shading, edges and alignment) and the linewidth (which supplies the
    Linewidth0 baseline).

    Each scan carries a "tab" dict holding (params, curve) for every tabletop
    signal. The baseline-subtracted signals reuse their parent's fit with the
    baseline shifted by the same constant - subtracting a constant from the
    data moves the tabletop down by exactly that amount and changes nothing
    else, so this is the same curve, not a second fit.
    """
    scans = _load_line_scan_folder_robust(folder_path)
    scans = bp.sum_normalize_line_scans(
        bp.fold_line_scans(scans, max_shift_ghz=8))
    results = []
    for s in scans:
        _, params = bf.fit_all_spectra(s["shift_freq"], s["folded_spectra_norm"])
        _, shifts, linewidths, _ = bf.extract_fit_parameters(params)
        spec_norm = s["folded_spectra_norm"]

        try:
            tab_params, tab_curve = bf.fit_tabletop(s["positions"], shifts)
            region = bf.tabletop_regions(s["positions"], tab_params, edge_pad=2.0)
        except Exception:
            tab_params, tab_curve, region = None, None, None

        try:
            tab_params_lw, tab_curve_lw = bf.fit_tabletop(
                s["positions"], linewidths, allow_negative=True)
        except Exception:
            tab_params_lw, tab_curve_lw = None, None

        bfs0 = _media_level(shifts, tab_params, region)
        lw0 = _media_level(linewidths, tab_params_lw, region)

        tab = {"bfs": (tab_params, tab_curve),
               "lw":  (tab_params_lw, tab_curve_lw)}
        for key, (parent, level) in (("bfs_0", ("bfs", bfs0)),
                                     ("lw_0", ("lw", lw0))):
            p, curve = tab[parent]
            if p is None or level is None:
                tab[key] = (None, None)
                continue
            # Scale the y-valued parameters too; x_left/x_right/slope_width
            # are positions in um and must not be touched.
            scale = MHZ_PER_GHZ if key in MHZ_SIGNALS else 1.0
            shifted = dict(p)
            shifted["baseline"] = (p["baseline"] - level) * scale
            shifted["amplitude"] = p["amplitude"] * scale
            tab[key] = (shifted,
                        None if curve is None else (curve - level) * scale)

        results.append({
            "tab":           tab,
            "scan":          s,
            "bfs":           shifts,
            "lw":            linewidths,
            "bfs_0":         None if bfs0 is None
                             else (shifts - bfs0) * MHZ_PER_GHZ,
            "lw_0":          None if lw0 is None
                             else (linewidths - lw0) * MHZ_PER_GHZ,
            "bfs0_level":    bfs0,
            "lw0_level":     lw0,
            "simple":        spectral_gradient_per_pixel(spec_norm, "simple"),
            "sobel":         spectral_gradient_per_pixel(spec_norm, "sobel"),
            "lap":           spectral_gradient_per_pixel(spec_norm, "lap"),
            "tab_params":    tab_params,
            "tab_curve":     tab_curve,
            "tab_params_lw": tab_params_lw,
            "region":        region,
        })
    return results


# ─────────────────────────────────────────────
# Region shading + centre alignment
# ─────────────────────────────────────────────

def _runs(mask):
    in_run, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            yield start, i
    if in_run:
        yield start, len(mask)


def shade_regions(ax, positions, region):
    if region is None:
        return
    for r, color in REGION_COLORS.items():
        for start, end in _runs(region == r):
            ax.axvspan(positions[start],
                       positions[min(end, len(positions) - 1)],
                       color=color, alpha=0.5, zorder=0)


def _aligned(r, signal_key):
    """(positions shifted so the tabletop centre is at 0, signal).

    Alignment always uses the SHIFT tabletop so every signal shares one
    geometry. Returns (None, None) when this scan has no usable fit, so the
    caller can skip it rather than silently mixing it in unaligned.
    """
    tp = r["tab_params"]
    sig = r.get(signal_key)
    if tp is None or sig is None:
        return None, None
    centre = 0.5 * (tp["x_left"] + tp["x_right"])
    positions = np.asarray(r["scan"]["positions"], dtype=float) - centre
    return positions, np.asarray(sig, dtype=float)


def _pooled(results, signal_key):
    """Every scan's aligned points concatenated into one dataset, sorted by x.

    Returns (x, y, n_used, n_skipped).
    """
    xs, ys, used, skipped = [], [], 0, 0
    for r in results:
        p, s = _aligned(r, signal_key)
        if p is None:
            skipped += 1
            continue
        xs.append(p)
        ys.append(s)
        used += 1
    if not xs:
        return None, None, 0, skipped
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = np.argsort(x)
    return x[order], y[order], used, skipped


def _aligned_mean(results, signal_key, n_grid=120, min_frac=None):
    """Point-wise mean across samples on a common centre-aligned grid.

    Centring shifts every scan by its own fitted midpoint, which is a float,
    so no two scans share x values and the mean cannot be taken column-wise.
    Each scan is therefore resampled onto a shared grid first.

    Grid points reached by fewer than `min_frac` of the folder's scans are
    dropped. Without that, the ends of the line are the average of a single
    scan while the middle averages all of them, which is what produced the
    ragged tails on the original chart.

    Returns (grid, mean, sd, n_per_point, n_scans).
    """
    if min_frac is None:
        min_frac = MEAN_MIN_FRAC
    curves = []
    for r in results:
        p, s = _aligned(r, signal_key)
        if p is None:
            continue
        finite = np.isfinite(p) & np.isfinite(s)
        if finite.sum() < 2:
            continue
        p, s = p[finite], s[finite]
        order = np.argsort(p)
        curves.append((p[order], s[order]))
    if not curves:
        return None, None, None, None, 0

    lo = min(float(p.min()) for p, _ in curves)
    hi = max(float(p.max()) for p, _ in curves)
    grid = np.linspace(lo, hi, n_grid)
    stack = np.vstack([np.interp(grid, p, s, left=np.nan, right=np.nan)
                       for p, s in curves])

    n_per_point = np.sum(np.isfinite(stack), axis=0)
    need = max(2, int(np.ceil(min_frac * len(curves))))
    keep = n_per_point >= need
    if not keep.any():
        return None, None, None, None, len(curves)

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
    return (grid[keep], mean[keep], sd[keep],
            n_per_point[keep], len(curves))


def _fit_pooled(x, y, signal_key):
    """Refit a single tabletop to the pooled point cloud."""
    try:
        return bf.fit_tabletop(x, y,
                               allow_negative=signal_key in NEGATIVE_OK_SIGNALS)[0]
    except Exception as exc:
        print(f"  [warn] pooled tabletop fit failed for {signal_key}: {exc}")
        return None


def _binned_mean(x, y, nbins=60):
    """Mean of the pooled cloud in equal-width x bins (used as the average
    line for signals that have no tabletop shape)."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return None, None
    x, y = x[finite], y[finite]
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            means[b] = np.mean(y[m])
    ok = np.isfinite(means)
    return centres[ok], means[ok]


def _signal_unit(signal_key):
    """The y-axis unit of a signal, read off its label."""
    label = SIGNAL_LABELS.get(signal_key, "")
    if "(MHz)" in label:
        return "MHz"
    if "(GHz)" in label:
        return "GHz"
    return ""


def _fmt_y(value, unit):
    """Format a y-valued quantity with a sensible number of decimals."""
    return f"{value:.1f}" if unit == "MHz" else f"{value:.3f}"


def _residual_units(signal_key):
    """(unit label, d^2 scale, RMS scale).

    Residuals are ~0.02 GHz, so GHz signals are reported in MHz to avoid
    leading zeros. Signals already stored in MHz need no scaling.
    """
    unit = _signal_unit(signal_key)
    if unit == "MHz":
        return "MHz", 1.0, 1.0
    if unit == "GHz":
        return "MHz", 1e6, 1e3
    return "units", 1.0, 1.0


def _scan_tabletop_residual(r, signal_key):
    """Mean squared distance from one scan's points to that scan's OWN
    tabletop fit.

    Returns (mean_d2, n_points), or (None, 0) if the scan has no signal or the
    fit failed.
    """
    sig = r.get(signal_key)
    if sig is None:
        return None, 0
    x = np.asarray(r["scan"]["positions"], dtype=float)
    y = np.asarray(sig, dtype=float)
    try:
        _, fitted = bf.fit_tabletop(
            x, y, allow_negative=signal_key in NEGATIVE_OK_SIGNALS)
    except Exception:
        return None, 0
    resid = y - fitted
    ok = np.isfinite(resid)
    if not ok.any():
        return None, 0
    return float(np.mean(resid[ok] ** 2)), int(ok.sum())


def _folder_mean_residual(results, signal_key):
    """Mean of the individual per-scan tabletop variances for one folder.

    This is the average of each scan fitted to itself, NOT the scatter about
    the pooled fit - it measures how tightly a single line scan follows a
    tabletop, independent of how much the cells differ from each other.
    """
    vals = [m for m, _ in (_scan_tabletop_residual(r, signal_key)
                           for r in results) if m is not None]
    return float(np.mean(vals)) if vals else None


# ─────────────────────────────────────────────
# 1) Per-folder grid
# ─────────────────────────────────────────────

def plot_folder_grid(results, folder_label, signal_key, ncols=3):
    ylabel = SIGNAL_LABELS[signal_key]
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 3.4),
                             squeeze=False)
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        sig = r[signal_key]
        if sig is None:
            ax.axis("off")
            continue
        if signal_key in TABLETOP_SIGNALS:
            shade_regions(ax, positions, r["region"])
        ax.plot(positions, sig, 'k.', markersize=4)
        if signal_key in TABLETOP_SIGNALS:
            # This signal's own tabletop, not always the shift one.
            p, curve = r["tab"].get(signal_key, (None, None))
            if curve is not None:
                ax.plot(positions, curve, 'r-', linewidth=1.8)
            if p is not None:
                ax.axvline(p["x_left"], color='r', linestyle='--', alpha=0.5)
                ax.axvline(p["x_right"], color='r', linestyle='--', alpha=0.5)
        ax.set_title(os.path.splitext(r["scan"]["filename"])[0], fontsize=11)
        ax.set_xlabel("Position (µm)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        _style_axes(ax)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{DRUG} {_flabel(folder_label)}", fontsize=16,
                 fontweight="bold")
    _finish(fig, f"{signal_key}_grid_{folder_label}")


# ─────────────────────────────────────────────
# 2) Per-folder overlay (centre-aligned)
# ─────────────────────────────────────────────

def plot_folder_overlay(results, folder_label, color, signal_key,
                        ax=None, show_title=True):
    ylabel = SIGNAL_LABELS[signal_key]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4.8))
    used = 0
    for r in results:
        positions, sig = _aligned(r, signal_key)
        if positions is None:
            continue
        used += 1
        ax.plot(positions, sig, 'o-', color=color,
                markersize=4, linewidth=1.0, alpha=0.85)
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
    if show_title:
        ax.set_title(f"{DRUG} {_flabel(folder_label)} (n={used})")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _style_axes(ax)
    if standalone:
        _finish(fig, f"{signal_key}_overlay_{folder_label}")


# ─────────────────────────────────────────────
# 3) All folders overlaid on ONE axes
# ─────────────────────────────────────────────

def plot_all_stacked(all_results, signal_key):
    ylabel = SIGNAL_LABELS[signal_key]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for folder in FOLDERS:
        color = FOLDER_COLORS[folder]
        first = True
        for r in all_results[folder]:
            positions, sig = _aligned(r, signal_key)
            if positions is None:
                continue
            ax.plot(positions, sig, 'o-', color=color,
                    markersize=4, linewidth=1.0, alpha=0.75,
                    label=_flabel(folder) if first else None)
            first = False
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.set_title(f"{DRUG} dose series - {ylabel}")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title="dose", loc="best")
    _style_axes(ax)
    _finish(fig, f"{signal_key}_all_stacked")


# ─────────────────────────────────────────────
# 4) Average line chart: the point-wise mean across samples
# ─────────────────────────────────────────────

def _draw_folder_means(ax, all_results, signal_key, report=False):
    """Draw one mean line per folder onto `ax`. Returns True if anything drew."""
    drew = False
    for folder in FOLDERS:
        results = all_results[folder]
        grid, mean, sd, npp, n = _aligned_mean(results, signal_key)
        if grid is None:
            print(f"  [warn] {folder}: no scans usable for '{signal_key}'")
            continue
        skipped = len(results) - n
        if skipped and report:
            print(f"  [warn] {folder}: {skipped} scan(s) skipped "
                  f"(no tabletop fit, so no centre to align on)")
        color = FOLDER_COLORS[folder]
        if MEAN_SHOW_SD:
            ax.fill_between(grid, mean - sd, mean + sd,
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(grid, mean, 'o-', color=color, markersize=4,
                linewidth=1.5, alpha=0.9, label=f"{_flabel(folder)} (n={n})")
        drew = True
        if report:
            print(f"  {folder:<10} n={n}  x {grid.min():7.1f} .. {grid.max():7.1f} um"
                  f"  ({grid.max() - grid.min():.0f} um)"
                  f"  scans per point {npp.min()}-{npp.max()}")
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
    return drew


def plot_folder_means(all_results, signal_key):
    """One connected mean line per concentration - the average line chart."""
    ylabel = SIGNAL_LABELS[signal_key]
    print(f"\nAverage line chart coverage - {_ascii(ylabel)}")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    _draw_folder_means(ax, all_results, signal_key, report=True)
    ax.set_title(f"{DRUG} dose series - average {ylabel}")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title="dose", loc="best")
    _style_axes(ax)
    _finish(fig, f"{signal_key}_average")


def plot_mean_panels(all_results, signal_keys=("bfs_0", "lw_0")):
    """The baseline-corrected averages in one figure, concentrations overlaid.

    Only BFS-BFS0 and LW-LW0 are shown: subtracting each scan's own media
    level removes the per-scan offset, so the doses are directly comparable in
    a way the raw BFS and linewidth are not.
    """
    keys = [k for k in signal_keys if k in SIGNAL_LABELS]
    ncols = min(2, len(keys)) or 1
    nrows = (len(keys) + ncols - 1) // ncols
    # Reserve a fixed strip for the shared legend rather than a fixed
    # fraction, so it does not eat the x labels when there is only one row.
    legend_in = 0.8
    fig_h = nrows * 4.6 + legend_in
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7.0, fig_h),
                             squeeze=False)
    axes = axes.flatten()
    handles, labels = [], []
    for i, key in enumerate(keys):
        ax = axes[i]
        _draw_folder_means(ax, all_results, key)
        ax.set_title(SIGNAL_LABELS[key], fontsize=13)
        ax.set_xlabel(POS_LABEL)
        ax.set_ylabel(SIGNAL_LABELS[key])
        ax.grid(alpha=0.3)
        _style_axes(ax)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{DRUG} dose series - average line charts",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1))
    if handles:
        fig.legend(handles, labels, title="dose", loc="lower center",
                   ncol=min(len(labels), 5), frameon=True)
    if SAVE_FIGS:
        outdir = os.path.join("figures", DRUG)
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "average_panels.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 5) Table tops: one panel per folder
# ─────────────────────────────────────────────

def plot_folder_tabletops(all_results, signal_key, ncols=2):
    """One table top per concentration, fitted to that folder's pooled points.

    The grey band is the medium (outside the cell); the dashed grey line is
    its level - the BFS0 / Linewidth0 that 'bfs_0' and 'linewidth_0' subtract.
    """
    ylabel = SIGNAL_LABELS[signal_key]
    n = len(FOLDERS)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 6.8, nrows * 4.7),
                             squeeze=False)
    axes = axes.flatten()
    for i, folder in enumerate(FOLDERS):
        ax = axes[i]
        x, y, used, skipped = _pooled(all_results[folder], signal_key)
        if x is None:
            ax.set_title(f"{_flabel(folder)} - no data")
            ax.axis("off")
            continue
        p = _fit_pooled(x, y, signal_key)
        if p is not None:
            pad = 2.0 * p["slope_width"]
            grey = REGION_COLORS["outside"]
            ax.axvspan(x.min(), max(p["x_left"] - pad, x.min()),
                       color=grey, alpha=0.9, zorder=0)
            ax.axvspan(min(p["x_right"] + pad, x.max()), x.max(),
                       color=grey, alpha=0.9, zorder=0)

        ax.plot(x, y, '.', color=FOLDER_COLORS[folder],
                markersize=4, alpha=0.45, zorder=2)

        if p is not None:
            grid = np.linspace(x.min(), x.max(), 500)
            ax.plot(grid, bf.tabletop(grid, **p), 'r-', linewidth=2.6, zorder=3)
            ax.axvline(p["x_left"], color='r', linestyle='--', alpha=0.6)
            ax.axvline(p["x_right"], color='r', linestyle='--', alpha=0.6)
            ax.axhline(p["baseline"], color='dimgrey', linestyle='--',
                       linewidth=1.6, zorder=3)
            unit_y = _signal_unit(signal_key)
            ax.annotate(f"media = {_fmt_y(p['baseline'], unit_y)} {unit_y}",
                        xy=(0.02, 0.06), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", color="dimgrey")
            width = p["x_right"] - p["x_left"]
            # Mean of the individual per-scan variances, matching the console
            # report - not the scatter about this pooled curve.
            unit, s_sq, _ = _residual_units(signal_key)
            mres = _folder_mean_residual(all_results[folder], signal_key)
            res_txt = ("" if mres is None
                       else f"\nmean d^2 = {mres * s_sq:.0f} {unit}^2")
            ax.set_title(f"{_flabel(folder)} (n={used})\n"
                         f"width = {width:.1f} µm   "
                         f"height = {_fmt_y(p['amplitude'], unit_y)} "
                         f"{unit_y}{res_txt}",
                         fontsize=12)
        else:
            ax.set_title(f"{_flabel(folder)} (n={used})", fontsize=12)
        if skipped:
            print(f"  [warn] {folder}: {skipped} scan(s) skipped "
                  f"(no tabletop fit)")
        ax.set_xlabel(POS_LABEL)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        _style_axes(ax)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{DRUG} table tops - {ylabel}", fontsize=16,
                 fontweight="bold")
    _finish(fig, f"{signal_key}_tabletops")


# ─────────────────────────────────────────────
# Scatter of each scan about its OWN tabletop fit
# ─────────────────────────────────────────────

def report_tabletop_residuals(all_results, signal_key):
    """Print the scatter of every scan about its own tabletop, then average.

    Each scan is fitted separately and 'mean d^2' is the average squared
    distance from that scan's points to its own curve; RMS is the square root,
    so it reads in the same units as the y axis. The per-dose figure is the
    mean of the individual scan variances - it measures how tightly a single
    line scan follows a tabletop, independent of how much the cells differ
    from each other.
    """
    if signal_key not in TABLETOP_SIGNALS:
        return
    unit, s_sq, s_lin = _residual_units(signal_key)
    print(f"\nPer-scan tabletop scatter - "
          f"{_ascii(SIGNAL_LABELS[signal_key])}")
    for folder in FOLDERS:
        print(f"  {folder}")
        vals = []
        for r in all_results[folder]:
            msq, npts = _scan_tabletop_residual(r, signal_key)
            name = os.path.splitext(r["scan"]["filename"])[0][:38]
            if msq is None:
                print(f"    {name:<40} {'no fit':>10}")
                continue
            vals.append(msq)
            print(f"    {name:<40} {npts:>4} pts  "
                  f"d^2 = {msq * s_sq:>8.1f}  RMS = {np.sqrt(msq) * s_lin:>6.2f}")
        if vals:
            v = np.array(vals)
            print(f"    {'-> mean over ' + str(len(v)) + ' scans':<40} "
                  f"          d^2 = {v.mean() * s_sq:>8.1f}"
                  f"  RMS = {np.sqrt(v.mean()) * s_lin:>6.2f}"
                  f"   (spread of d^2 +/- {v.std() * s_sq:.1f})")
        else:
            print(f"    {'-> no usable scans':<40}")
    print(f"  (d^2 in {unit}^2, RMS in {unit})")


# ─────────────────────────────────────────────
# Cross-check against the vendor's own fit (xlsx only)
# ─────────────────────────────────────────────

# Physically plausible Brillouin shift window for a cell in medium; vendor
# values outside it are diverged fits, not real measurements.
PLAUSIBLE_BFS_GHZ = (4.0, 9.0)


def report_vendor_agreement(all_results):
    """Compare our refitted BFS/linewidth with the values the acquisition
    software stored in the '*_Fitted.xlsx' files.

    This is a sanity check on the shared pipeline, not a correction: the
    vendor's stored fits diverge on some pixels (values in the hundreds of
    GHz), which is why BFS and linewidth are refit from the raw spectra for
    every file regardless of format.
    """
    rows, bad, total = [], 0, 0
    for folder in FOLDERS:
        for r in all_results.get(folder, []):
            s = r["scan"]
            if s.get("vendor_bfs") is None:
                continue
            vb, vl = s["vendor_bfs"], s["vendor_lw"]
            ok = (np.isfinite(vb) & np.isfinite(r["bfs"])
                  & (vb >= PLAUSIBLE_BFS_GHZ[0]) & (vb <= PLAUSIBLE_BFS_GHZ[1]))
            total += vb.size
            bad += int((~ok).sum())
            if ok.sum() < 3:
                continue
            rows.append((s["filename"], r["bfs"][ok], vb[ok],
                         r["lw"][ok], None if vl is None else vl[ok]))
    if not rows:
        return

    print("\nCross-check vs vendor Lorentzian fit (xlsx files):")
    print(f"  {bad}/{total} vendor points fell outside "
          f"{PLAUSIBLE_BFS_GHZ[0]}-{PLAUSIBLE_BFS_GHZ[1]} GHz "
          f"(diverged vendor fits, excluded below)")
    corr = [np.corrcoef(a, b)[0, 1] for _, a, b, _, _ in rows]
    bias = [np.median(a - b) for _, a, b, _, _ in rows]
    print(f"  BFS: correlation per scan min {min(corr):.3f}, "
          f"median {np.median(corr):.3f}; "
          f"median offset {np.median(bias):+.4f} GHz")
    ratios = [np.median(d / c) for _, _, _, c, d in rows if d is not None]
    if ratios:
        print(f"  linewidth: vendor/ours ratio median {np.median(ratios):.3f} "
              f"(ours is the Lorentzian half-width gamma)")
    worst = min(zip(corr, (r[0] for r in rows)))
    if worst[0] < 0.8:
        print(f"  [warn] weakest agreement: {worst[1]} (corr {worst[0]:.3f}) "
              f"- check the vendor fit for that file")


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def make_all_plots(all_results, signal_key):
    for folder in FOLDERS:
        plot_folder_grid(all_results[folder], folder, signal_key, ncols=3)
    for folder in FOLDERS:
        plot_folder_overlay(all_results[folder], folder,
                            FOLDER_COLORS[folder], signal_key)
    plot_all_stacked(all_results, signal_key)
    plot_folder_means(all_results, signal_key)
    if signal_key in TABLETOP_SIGNALS:
        report_tabletop_residuals(all_results, signal_key)
        plot_folder_tabletops(all_results, signal_key)
    # The baseline-corrected averages, regardless of SIGNAL_KIND.
    plot_mean_panels(all_results)


if __name__ == "__main__":
    # 'bfs', 'lw', 'bfs_0', 'lw_0', 'simple', 'sobel', 'lap'.
    SIGNAL_KIND = "lw_0"

    all_results = {}
    for folder in FOLDERS:
        path = os.path.join(BASE, folder)
        print(f"Loading {path} ...")
        all_results[folder] = load_and_analyse(path)
        print(f"  {len(all_results[folder])} scans")

    report_vendor_agreement(all_results)
    make_all_plots(all_results, SIGNAL_KIND)
