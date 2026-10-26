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
  6. Dose lines (tabletop modes only): two one-point-per-dose charts -
     periphery % and mean d^2.
  7. Summary: one figure with the bfs_0, lw_0 and sobel average line charts
     plus BFS mean d^2 vs dose, drawn every run regardless of SIGNAL_KIND.

Scans are combined by shifting each one so the centre of its fitted tabletop
sits at x = 0, then pooling every point into a single dataset and refitting
one tabletop. Absolute stage position is discarded; cells of differing size
are simply centred on each other.

The plotted quantity is chosen by SIGNAL_KIND at the bottom of the file.
Five tabletop modes, which get a fitted tabletop and region shading:
  'bfs'   - Lorentzian-fit Brillouin frequency shift (GHz), stiffness
  'lw'    - Lorentzian-fit Brillouin linewidth (GHz), viscosity
  'bfs_0' - BFS minus that scan's tabletop baseline (MHz)
  'lw_0'  - linewidth minus its tabletop baseline (MHz)
  'vca'   - Vertex Component Analysis with 2 endmembers per scan, the cell
            endmember's abundance (0-1)

One gradient mode, restricted to the CENTRE region of the scan's own shift
tabletop so it describes texture inside the spheroid rather than the
boundary. It is annotated with max and mean and shares one y scale across
doses:
  'sobel' - Sobel across pixels on the raw folded spectrum; responds to ANY
            change in the spectrum

Only Sobel is retained of the three kernels - its 5-pixel span smooths noise
where the 2- and 3-pixel [-1,1] and Laplacian kernels amplified it.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd


# ─────────────────────────────────────────────
# Dataset config - uncomment ONE block
# ─────────────────────────────────────────────

# Stage step between adjacent pixels. The X (csv) and Z (xlsx) columns hold
# step INDICES, not microns, so positions are multiplied by this to get um.
STEP_UM = 5.0

# Untreated spheroids are the same sample for both drugs, so both dose series
# share one control group from this folder rather than each keeping its own.
CONTROL_DIR = "Spheroid Controls"

# --- PACLITAXEL ---------------------------------------------------------

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
FOLDER_PATHS  = {"Control": CONTROL_DIR}
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
# Anything not listed here is read from BASE/<folder> as usual.
FOLDER_PATHS = {"control": CONTROL_DIR}
'''


# Folder names are directory names; these are what appears on the charts.
# uM here is micromolar (concentration) - not to be confused with the um in
# the position axis, which is micrometres.
FOLDER_LABELS = {
    "control": "Control",
    "100um":   "100 µM",
    "200um":   "200 µM",
    "400um":   "400 µM",
}


# Where every generated figure is written, as well as being shown.
# Set to None to disable saving entirely.
# Follows whichever config block is active above, so carboplatin figures land
# in 'Brillouin Images/Carboplatin' and paclitaxel in
# 'Brillouin Images/Paclitaxel'. Relative to the working directory, which must
# be the repo root (BASE is relative too).
SAVE_PATH = os.path.join("Brillouin Images", DRUG)
# SAVE_PATH = None

REGION_COLORS = {"outside": "#dddddd", "edge": "#ffd9b3", "center": "#b3e6ff"}

SIGNAL_LABELS = {
    "bfs":       "Brillouin shift (GHz)",
    "lw":        "Brillouin linewidth (GHz)",
    "bfs_0":     "BFS - BFS₀ (MHz)",
    "lw_0":      "LW - LW₀ (MHz)",
    "vca":       "VCA abundance (cell endmember)",
    "sobel":     "|spectral gradient| - Sobel [1, 2, 0, -2, -1]",
}

# Endmembers per line scan for the VCA modes. With 2, the abundances sum to 1
# per pixel, so the two columns are exact complements and the profile has a
# single degree of freedom.
VCA_ENDMEMBERS = 2

# The baseline-corrected signals are only a few hundred MHz, so they are
# stored and plotted in MHz rather than GHz (1 GHz = 1000 MHz). Converting at
# the source means the axis, the fitted height and the variance are all in the
# same unit with no per-plot rescaling. Raw bfs/lw stay in GHz - 5.69 GHz
# would read as an unwieldy 5690 MHz.
MHZ_PER_GHZ = 1000.0
MHZ_SIGNALS = {"bfs_0", "lw_0"}

# TFP calibration correction (Irina/Hadi, via Ashutosh): every fitted BFS and
# linewidth is multiplied by this. Applied once at the fit, so BFS0/LW0, SDs
# and derivatives all inherit it. Tabletop fits are scale-equivariant, so the
# region geometry is unchanged. Set to 1.0 for the uncorrected values.
TFP_CALIBRATION = 1.069348

# Signals that have a meaningful tabletop (media -> cell -> media) shape.
# The gradient modes are excluded: they spike at the two cell edges rather
# than forming a plateau, so a tabletop fit would not mean anything.
TABLETOP_SIGNALS = {"bfs", "lw", "bfs_0", "lw_0", "vca"}

# Signals whose plateau may sit BELOW the surrounding medium, so the tabletop
# fit must be allowed a negative amplitude.
NEGATIVE_OK_SIGNALS = {"lw", "lw_0"}

# Gradient modes. These are restricted to the CENTRE region of the scan's own
# shift tabletop, so they measure texture inside the cell rather than the
# cell-edge transition, and their charts are annotated with max and mean.
# Only the Sobel kernel is kept: its 5-pixel span smooths per-pixel noise,
# where the 2- and 3-pixel [-1,1] and Laplacian kernels amplified it (measured
# on the VCA family, edge-vs-media response was 0.095 vs 0.094 - no contrast).
GRADIENT_SIGNALS = {"sobel"}

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


def folder_path(folder):
    """Directory holding a dose group's scans.

    Normally BASE/<folder>, but FOLDER_PATHS can point a group somewhere else
    entirely - used so both drugs read their controls from the one shared
    spheroid-control folder.
    """
    return FOLDER_PATHS.get(folder, os.path.join(BASE, folder))


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


def _save(fig, name, subdir=None):
    """Write a figure to SAVE_PATH, unless saving is switched off.

    Per-mode figures pass subdir=<signal_key> so each mode gets its own
    folder; the cross-mode summary passes nothing and lands alongside them.
    """
    if not SAVE_PATH:
        return
    outdir = SAVE_PATH if subdir is None else os.path.join(SAVE_PATH, subdir)
    os.makedirs(outdir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    fig.savefig(os.path.join(outdir, safe + ".png"),
                dpi=300, bbox_inches="tight")


def _finish(fig, name, tight=True, subdir=None):
    """Tidy layout, save, then show.

    Pass tight=False when the caller has already run tight_layout with a
    reserved strip (e.g. for a legend below the axes); running it again
    without the rect would reclaim that space and overlap the legend.
    """
    if tight:
        fig.tight_layout()
    _save(fig, name, subdir)
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
    return _scans_from_raw_table(pd.read_csv(full), fname)


def _scans_from_raw_table(df, fname):
    """Split an acquisition table into one scan per sweep.

    The table is the 13-column layout the instrument writes - X/Y/Z, a
    comma-joined 'Raw Data' spectrum per row, and the frequency window. It
    arrives either as a .csv or, for some scans, as a one-sheet .xlsx; both
    land here so the two formats cannot drift apart.
    """
    out = []
    lengths = df["Raw Data"].apply(lambda s: len(str(s).split(","))).to_numpy()
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
                    lambda s: np.array([float(v) for v in str(s).split(",")]))
            )
            start = sub_pass["startFrequency"].iloc[0]
            end   = sub_pass["endFrequency"].iloc[0]
            freq  = np.linspace(start, end, spectra.shape[1])
            out.append(_sorted_scan(spectra, freq,
                                    sub_pass[axis].to_numpy(), fname))
    return out


def _load_workbook_scan(full, fname):
    """Read a scan from a workbook (.xlsx or legacy .xls).

    '*_Fitted.xlsx' from the acquisition software has three sheets ('Fitted',
    'FittingData_spectrum', 'Raw Data'). Other scans are saved as the plain
    acquisition table in a single sheet - the same columns as a CSV. Dispatch
    on the sheet names rather than the filename, since the naming is not
    consistent.
    """
    try:
        sheets = pd.ExcelFile(full).sheet_names
    except Exception as exc:
        print(f"  [warn] {fname}: cannot open ({exc})")
        return []
    if "Raw Data" in sheets and "Fitted" in sheets:
        return _load_xlsx_fitted(full, fname)
    df = pd.read_excel(full, sheet_name=0)
    if "Raw Data" not in df.columns:
        print(f"  [warn] {fname}: no recognisable spectra "
              f"(sheets: {sheets}) - skipped")
        return []
    return _scans_from_raw_table(df, fname)


def _load_xlsx_fitted(full, fname):
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


# Scans arrive as CSV, as modern .xlsx, and as legacy .xls - all three hold
# the same acquisition table. Reading .xls needs the `xlrd` package.
SPECTRA_EXTS = (".csv", ".xlsx", ".xls")

# Files that legitimately sit alongside the data and should pass without
# comment. Anything else gets a warning rather than being dropped in silence.
NON_DATA_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf",
                 ".txt", ".md", ".ini", ".ds_store")


def _load_line_scan_folder_robust(folder_path):
    out = []
    for fname in sorted(os.listdir(folder_path)):
        low = fname.lower()
        full = os.path.join(folder_path, fname)
        # Excel lock files, dotfiles and subdirectories are never data.
        if fname.startswith("~$") or fname.startswith(".") \
                or os.path.isdir(full):
            continue
        if low.endswith(".csv"):
            out.extend(_load_csv_scans(full, fname))
        elif low.endswith((".xlsx", ".xls")):
            out.extend(_load_workbook_scan(full, fname))
        elif not low.endswith(NON_DATA_EXTS):
            print(f"  [warn] {fname}: unrecognised file type, not loaded - "
                  f"expected one of {SPECTRA_EXTS}")
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


def profile_gradient(sig, kind):
    """Apply a 1D kernel ALONG the position axis; |response| per pixel.

    Distinct from spectral_gradient_per_pixel, which sums differences across a
    whole spectrum. This one works on a scalar profile such as the VCA
    abundance. NaN where the kernel does not fit at the ends.

    Currently unused - the profile-gradient modes it fed (bfs_*, lw_*, vca_*)
    were pruned as noise-dominated. Kept because modes are still being added
    and removed and this would only need rewriting.
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    out = np.full(n, np.nan)
    if kind == "simple":       # [-1, 1] forward difference
        for i in range(n - 1):
            out[i] = abs(sig[i + 1] - sig[i])
    elif kind == "lap":        # [1, -2, 1]
        for i in range(1, n - 1):
            out[i] = abs(sig[i - 1] - 2.0 * sig[i] + sig[i + 1])
    elif kind == "sobel":      # [1, 2, 0, -2, -1]
        for i in range(2, n - 2):
            out[i] = abs(sig[i - 2] + 2.0 * sig[i - 1]
                         - 2.0 * sig[i + 1] - sig[i + 2])
    else:
        raise ValueError(f"unknown gradient kind: {kind}")
    return out


def _cell_minus_media(sig, region):
    """How much higher a profile sits inside the cell than outside it.

    Uses the fitted region labels when available; those come from the shift
    tabletop and give the cell's real extent. Falls back to fixed slices of
    the scan (middle third vs outer fifths) when no fit succeeded, which
    assumes the cell is roughly centred.
    """
    sig = np.asarray(sig, dtype=float)
    if region is not None:
        centre = (region == "center") & np.isfinite(sig)
        outside = (region == "outside") & np.isfinite(sig)
        if centre.any() and outside.any():
            return float(np.mean(sig[centre]) - np.mean(sig[outside]))
    n = len(sig)
    if n < 5:
        return 0.0
    mid = sig[n // 3:2 * n // 3]
    edge = np.concatenate([sig[:n // 5], sig[4 * n // 5:]])
    if not np.isfinite(mid).any() or not np.isfinite(edge).any():
        return 0.0
    return float(np.nanmean(mid) - np.nanmean(edge))


def _vca_profile(spec_norm, region):
    """VCA(2) abundance for the cell endmember, oriented consistently.

    With 2 endmembers the abundance columns sum to 1 per pixel, so column 1 is
    exactly 1 - column 0 and "which endmember" is only a choice of
    orientation. VCA picks its endmembers by largest-magnitude projection, so
    the cell lands in column 0 for some scans and the medium for others; left
    uncorrected roughly half the scans would be inverted and dose averages
    would partly cancel. Flipping so the cell always reads high also lets
    fit_tabletop work, since it bounds amplitude >= 0 by default.
    """
    endmembers, _ = bd.vca(spec_norm, num_endmembers=VCA_ENDMEMBERS)
    ab = bd.estimate_abundances(spec_norm, endmembers)
    em = ab[:, 0]
    return 1.0 - em if _cell_minus_media(em, region) < 0 else em


def _ordered_edges(params):
    """Copy of a tabletop's params with x_left <= x_right.

    fit_tabletop bounds both edges to the scan range but never enforces their
    order, and with allow_negative=True the linewidth fit can settle on
    swapped edges with a negative amplitude - an equivalent curve, but
    tabletop_regions computes the centre as (x > x_left + pad) & (x < x_right
    - pad), which is empty when they are swapped. That would silently read as
    "100% border, no centre".
    """
    if params is None:
        return None
    out = dict(params)
    if out["x_left"] > out["x_right"]:
        out["x_left"], out["x_right"] = out["x_right"], out["x_left"]
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
        # fit_all_spectra also returns the fitted Lorentzian curves; nothing
        # needs them now that the fitted-curve gradient mode is gone.
        _, params = bf.fit_all_spectra(s["shift_freq"],
                                       s["folded_spectra_norm"])
        _, shifts, linewidths, _ = bf.extract_fit_parameters(params)
        shifts = np.asarray(shifts, dtype=float) * TFP_CALIBRATION
        linewidths = np.asarray(linewidths, dtype=float) * TFP_CALIBRATION
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

        # Regions from the LINEWIDTH tabletop, so the spheroid boundary can be
        # measured on viscosity as well as on stiffness. The BFS `region`
        # above is left alone - alignment, gradient masking and BFS0/LW0 all
        # share that one geometry.
        region_lw = None
        if tab_params_lw is not None:
            try:
                region_lw = bf.tabletop_regions(
                    s["positions"], _ordered_edges(tab_params_lw), edge_pad=2.0)
            except Exception:
                region_lw = None

        bfs0 = _media_level(shifts, tab_params, region)
        lw0 = _media_level(linewidths, tab_params_lw, region)

        # VCA(2) per scan. A failure here should cost only the VCA modes for
        # this scan, not the whole folder.
        try:
            vca_sig = _vca_profile(spec_norm, region)
        except Exception as exc:
            print(f"  [warn] {s['filename']}: VCA failed ({exc})")
            vca_sig = None
        if vca_sig is None:
            tab_params_vca, tab_curve_vca = None, None
        else:
            try:
                tab_params_vca, tab_curve_vca = bf.fit_tabletop(
                    s["positions"], vca_sig)
            except Exception:
                tab_params_vca, tab_curve_vca = None, None

        tab = {"bfs": (tab_params, tab_curve),
               "lw":  (tab_params_lw, tab_curve_lw),
               "vca": (tab_params_vca, tab_curve_vca)}
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

        def _centre_only(sig):
            """Blank a gradient outside the cell interior.

            The gradient modes are meant to describe texture WITHIN the cell,
            so everything outside the shift tabletop's centre region is set to
            NaN. Downstream code already drops non-finite points, so the
            restriction propagates to every figure. With no region info there
            is nothing to restrict against, so the profile is left whole.
            """
            if sig is None or region is None:
                return sig
            out = np.asarray(sig, dtype=float).copy()
            out[region != "center"] = np.nan
            return out

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
            # Sobel across pixels on the raw spectrum, summed over all bins.
            "sobel":         _centre_only(
                                 spectral_gradient_per_pixel(spec_norm, "sobel")),
            "vca":           vca_sig,
            "tab_params":    tab_params,
            "tab_curve":     tab_curve,
            "tab_params_lw": tab_params_lw,
            "region":        region,
            "region_lw":     region_lw,
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
    leading zeros. Signals already stored in MHz need no scaling. Dimensionless
    signals (VCA abundance, the gradients) get an empty unit label.
    """
    unit = _signal_unit(signal_key)
    if unit == "MHz":
        return "MHz", 1.0, 1.0
    if unit == "GHz":
        return "MHz", 1e6, 1e3
    return "", 1.0, 1.0


def _fmt_resid(value):
    """Format a residual readably at any scale.

    MHz residuals run to hundreds while dimensionless ones (VCA abundance) are
    ~0.01, so a fixed number of decimals renders one of them as '0'.
    """
    if not np.isfinite(value):
        return "n/a"
    a = abs(value)
    if a >= 100:
        return f"{value:.0f}"
    if a >= 1:
        return f"{value:.1f}"
    if a >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _unit_sq(unit):
    """' MHz^2' or '' for a dimensionless signal."""
    return f" {unit}^2" if unit else ""


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


def periphery_percent(results, region_key="region"):
    """Spheroid periphery as a percentage of the whole spheroid, per FILE.

        periphery % = edge / (edge + centre) * 100

    'edge' (orange on the shaded plots) and 'centre' (blue) are the regions
    bf.tabletop_regions assigns; 'outside' (grey) is the medium and plays no
    part. Counted in PIXELS, so the number matches exactly what is shaded,
    including where a scan ends before its edge zone does.

    region_key picks which boundary is measured:
        'region'     the BFS tabletop - how sharply STIFFNESS transitions
        'region_lw'  the LW tabletop  - how sharply VISCOSITY transitions

    Note there is no VCA region, so a vca panel reports the BFS periphery.

    A sharply delimited spheroid has a thin periphery and scores low; one whose
    boundary fades into the medium scores high. A scan whose fit leaves no
    plateau at all scores 100%.
    """
    by_file = {}
    for r in results:
        region = r.get(region_key)
        if region is None:
            continue
        n_edge = int((region == "edge").sum())
        n_centre = int((region == "center").sum())
        if n_edge + n_centre == 0:
            continue
        by_file.setdefault(r["scan"]["filename"], []).append(
            n_edge / (n_edge + n_centre) * 100.0)
    return np.array([np.mean(v) for v in by_file.values()])


def _periphery_region_key(signal_key):
    """Which regions a signal's periphery should be counted from."""
    return "region_lw" if signal_key in ("lw", "lw_0") else "region"


def mean_scan_periphery(results, region_key="region"):
    """Mean periphery over SCANS - the average of the individual tabletops.

    This is what the pooled tabletop panel shows, so the number there is
    exactly the average of the per-scan numbers on the grid panels.

    Differs from periphery_percent, which averages the passes of a file
    together first: a two-pass CSV is one spheroid measured twice, so for
    STATISTICS it must count once. Both views are wanted - matching the plots
    here, avoiding pseudo-replication there - so they are separate functions.
    The two agree wherever every file is single-pass.
    """
    vals = [_scan_periphery(r, region_key) for r in results]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _scan_periphery(r, region_key="region"):
    """One scan's periphery %, or None if it has no usable regions."""
    region = r.get(region_key)
    if region is None:
        return None
    n_edge = int((region == "edge").sum())
    n_centre = int((region == "center").sum())
    if n_edge + n_centre == 0:
        return None
    return n_edge / (n_edge + n_centre) * 100.0


def _folder_mean_residual(results, signal_key):
    """Mean of the individual per-scan tabletop variances for one folder.

    This is the average of each scan fitted to itself, NOT the scatter about
    the pooled fit - it measures how tightly a single line scan follows a
    tabletop, independent of how much the cells differ from each other.
    """
    vals = [m for m, _ in (_scan_tabletop_residual(r, signal_key)
                           for r in results) if m is not None]
    return float(np.mean(vals)) if vals else None


def _fmt_val(v):
    """Format a signal value readably at any scale."""
    if not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.001:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _stats_text(values):
    """'max = X   mean = Y' for a set of values, or None if there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    return f"max = {_fmt_val(v.max())}   mean = {_fmt_val(v.mean())}"


def _annotate_stats(ax, values, loc=(0.02, 0.94)):
    """Put max/mean in the corner of a gradient chart."""
    txt = _stats_text(values)
    if txt is None:
        return
    ax.annotate(txt, xy=loc, xycoords="axes fraction",
                fontsize=11, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.85))


def _folder_values(results, signal_key):
    """All finite values of a signal pooled across a folder's scans."""
    out = []
    for r in results:
        sig = r.get(signal_key)
        if sig is None:
            continue
        v = np.asarray(sig, dtype=float)
        out.append(v[np.isfinite(v)])
    return np.concatenate(out) if out else np.array([])


def _gradient_ylim(all_results, signal_key):
    """(0, global max) so every gradient panel shares one scale.

    Without this each panel autoscales and two doses with very different
    magnitudes look identical, which is exactly what the gradient charts are
    meant to distinguish. Returns None for non-gradient signals.
    """
    if signal_key not in GRADIENT_SIGNALS:
        return None
    vals = [_folder_values(all_results[f], signal_key) for f in FOLDERS]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    gmax = float(np.max(np.concatenate(vals)))
    if not np.isfinite(gmax) or gmax <= 0:
        return None
    return (0.0, gmax * 1.05)


# ─────────────────────────────────────────────
# 1) Per-folder grid
# ─────────────────────────────────────────────

def plot_folder_grid(results, folder_label, signal_key, ncols=3, ylim=None):
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
        title = os.path.splitext(r["scan"]["filename"])[0]
        if signal_key in TABLETOP_SIGNALS:
            # Shade, measure and outline all from the SAME regions - the
            # signal's own, so an lw panel is not shaded by the BFS boundary
            # while its curve and periphery describe the linewidth one.
            region_key = _periphery_region_key(signal_key)
            shade_regions(ax, positions, r.get(region_key))
        ax.plot(positions, sig, 'k.', markersize=4)
        if signal_key in TABLETOP_SIGNALS:
            # This signal's own tabletop, not always the shift one.
            p, curve = r["tab"].get(signal_key, (None, None))
            if curve is not None:
                ax.plot(positions, curve, 'r-', linewidth=1.8)
            if p is not None:
                ax.axvline(p["x_left"], color='r', linestyle='--', alpha=0.5)
                ax.axvline(p["x_right"], color='r', linestyle='--', alpha=0.5)
            periph = _scan_periphery(r, region_key)
            if periph is not None:
                title += f"\nperiphery = {periph:.1f}%"
        if signal_key in GRADIENT_SIGNALS:
            _annotate_stats(ax, sig)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Position (µm)")
        ax.grid(alpha=0.3)
        _style_axes(ax)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    # One figure-level y label: these are long enough that a per-panel label
    # collides with the panel above and below.
    fig.supylabel(ylabel, fontsize=13, fontweight="bold")
    fig.suptitle(f"{DRUG} {_flabel(folder_label)}", fontsize=16,
                 fontweight="bold")
    _finish(fig, f"{signal_key}_grid_{folder_label}", subdir=signal_key)


# ─────────────────────────────────────────────
# 2) Per-folder overlay (centre-aligned)
# ─────────────────────────────────────────────

def plot_folder_overlay(results, folder_label, color, signal_key,
                        ax=None, show_title=True, ylim=None):
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
    if signal_key in GRADIENT_SIGNALS:
        _annotate_stats(ax, _folder_values(results, signal_key))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_title:
        ax.set_title(f"{DRUG} {_flabel(folder_label)} (n={used})")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _style_axes(ax)
    if standalone:
        _finish(fig, f"{signal_key}_overlay_{folder_label}", subdir=signal_key)


# ─────────────────────────────────────────────
# 3) All folders overlaid on ONE axes
# ─────────────────────────────────────────────

def plot_all_stacked(all_results, signal_key):
    ylabel = SIGNAL_LABELS[signal_key]
    wide_legend = signal_key in GRADIENT_SIGNALS
    fig, ax = plt.subplots(figsize=(12, 7.0 if wide_legend else 5.8))
    for folder in FOLDERS:
        color = FOLDER_COLORS[folder]
        label = _flabel(folder)
        if signal_key in GRADIENT_SIGNALS:
            stats = _stats_text(_folder_values(all_results[folder], signal_key))
            if stats:
                label = f"{label}  ({stats})"
        first = True
        for r in all_results[folder]:
            positions, sig = _aligned(r, signal_key)
            if positions is None:
                continue
            ax.plot(positions, sig, 'o-', color=color,
                    markersize=4, linewidth=1.0, alpha=0.75,
                    label=label if first else None)
            first = False
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
    gylim = _gradient_ylim(all_results, signal_key)
    if gylim is not None:
        ax.set_ylim(*gylim)
    ax.set_title(f"{DRUG} dose series - {ylabel}")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if wide_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.legend(handles, labels, title="dose", loc="lower center",
                       ncol=2, frameon=True)
    else:
        ax.legend(title="dose", loc="best")
    _style_axes(ax)
    _finish(fig, f"{signal_key}_all_stacked", tight=not wide_legend,
            subdir=signal_key)


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
        label = f"{_flabel(folder)} (n={n})"
        if signal_key in GRADIENT_SIGNALS:
            # Stats of the mean line itself, i.e. of what is drawn here.
            stats = _stats_text(mean)
            if stats:
                label = f"{label}  {stats}"
        ax.plot(grid, mean, 'o-', color=color, markersize=4,
                linewidth=1.5, alpha=0.9, label=label)
        drew = True
        if report:
            print(f"  {folder:<10} n={n}  x {grid.min():7.1f} .. {grid.max():7.1f} um"
                  f"  ({grid.max() - grid.min():.0f} um)"
                  f"  scans per point {npp.min()}-{npp.max()}")
            gs = _stats_text(_folder_values(results, signal_key))
            if signal_key in GRADIENT_SIGNALS and gs:
                print(f"  {'':<10} centre-region points: {gs}")
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.2, alpha=0.8)
    return drew


def plot_folder_means(all_results, signal_key):
    """One connected mean line per concentration - the average line chart."""
    ylabel = SIGNAL_LABELS[signal_key]
    print(f"\nAverage line chart coverage - {_ascii(ylabel)}")
    # Gradient legends carry max/mean per dose and are far too wide to sit
    # over the data, so they go underneath the axes instead.
    wide_legend = signal_key in GRADIENT_SIGNALS
    fig, ax = plt.subplots(figsize=(12, 7.0 if wide_legend else 5.8))
    _draw_folder_means(ax, all_results, signal_key, report=True)
    if wide_legend:
        # Mean lines, so the raw global max would squash them; pin the floor
        # at 0 and let the top autoscale.
        ax.set_ylim(bottom=0.0)
    ax.set_title(f"{DRUG} dose series - average {ylabel}")
    ax.set_xlabel(POS_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if wide_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.legend(handles, labels, title="dose", loc="lower center",
                       ncol=2, frameon=True)
    else:
        ax.legend(title="dose", loc="best")
    _style_axes(ax)
    _finish(fig, f"{signal_key}_average", tight=not wide_legend,
            subdir=signal_key)


def _folder_scan_peripheries(results, region_key):
    """Every scan's periphery % in a folder, as an array."""
    vals = [_scan_periphery(r, region_key) for r in results]
    return np.array([v for v in vals if v is not None])


def _folder_scan_residuals(results, signal_key):
    """Every scan's mean squared distance from its OWN tabletop, as an array."""
    vals = [_scan_tabletop_residual(r, signal_key)[0] for r in results]
    return np.array([v for v in vals if v is not None])


def _draw_dose_series(ax, all_results, getter, ylabel, colour="tab:red",
                      scale=1.0, label=None, show_points=True):
    """One point per dose: mean over SCANS, +/- SEM, individuals behind.

    Averaging over scans (not files) keeps every point equal to the mean of
    the numbers printed on that dose's individual tabletop panels.

    Pass a label and show_points=False to overlay several series on one axes -
    individual dots from two series on top of each other are unreadable.
    """
    x = np.arange(len(FOLDERS))
    means, sems = [], []
    for i, folder in enumerate(FOLDERS):
        v = np.asarray(getter(all_results[folder]), dtype=float) * scale
        v = v[np.isfinite(v)]
        means.append(v.mean() if v.size else np.nan)
        sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
        if v.size and show_points:
            ax.plot(np.full(v.size, i), v, '.', color="0.6",
                    markersize=7, zorder=1)
    ax.errorbar(x, means, yerr=sems, marker='o', markersize=8, linewidth=2.2,
                capsize=5, color=colour, zorder=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([_flabel(f) for f in FOLDERS])
    ax.set_xlabel("dose")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _style_axes(ax)
    # The caller sets the y limits: set_ylim here would freeze autoscaling and
    # silently clip any series drawn onto the same axes afterwards.
    return np.array(means), np.array(sems)


def plot_tabletop_dose_lines(all_results, signal_key):
    """Two per-dose line charts for a tabletop mode.

    Left: periphery %, how much of the spheroid is boundary rather than
    plateau. Right: mean d^2, how far the points scatter from their own
    fitted tabletop - a measure of how tabletop-like the profile is at all.
    """
    if signal_key not in TABLETOP_SIGNALS:
        return
    region_key = _periphery_region_key(signal_key)
    unit, s_sq, _ = _residual_units(signal_key)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    _draw_dose_series(axes[0], all_results,
                      lambda res: _folder_scan_peripheries(res, region_key),
                      "periphery as % of whole spheroid", "tab:red")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Periphery vs dose", fontsize=13)

    m, s = _draw_dose_series(axes[1], all_results,
                             lambda res: _folder_scan_residuals(res, signal_key),
                             f"mean d^2{_unit_sq(unit)}", "tab:blue",
                             scale=s_sq)
    axes[1].set_ylim(0, np.nanmax(m + s) * 1.12)
    axes[1].set_title("Scatter about the tabletop vs dose", fontsize=13)

    fig.suptitle(f"{DRUG} - {SIGNAL_LABELS[signal_key]}",
                 fontsize=15, fontweight="bold")
    _finish(fig, f"{signal_key}_dose_lines", subdir=signal_key)


def plot_summary(all_results, line_keys=("bfs_0", "lw_0", "sobel")):
    """One figure gathering the dose series: three average line charts plus
    the periphery-vs-dose chart.

    Top row is the baseline-corrected shift and linewidth - subtracting each
    scan's own media level removes the per-scan offset, so doses are directly
    comparable in a way the raw BFS and linewidth are not. Bottom row pairs
    the spectral-gradient average (texture inside the spheroid) with how far
    the BFS and LW points scatter from their own tabletops at each dose.
    """
    keys = [k for k in line_keys if k in SIGNAL_LABELS]
    ncols = 2
    nrows = (len(keys) + 1 + ncols - 1) // ncols
    # Reserve a fixed strip for the shared legend rather than a fixed
    # fraction, so it does not eat the x labels.
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
        if key in GRADIENT_SIGNALS:
            ax.set_ylim(bottom=0.0)
        ax.set_title(SIGNAL_LABELS[key], fontsize=13)
        ax.set_xlabel(POS_LABEL)
        ax.set_ylabel(SIGNAL_LABELS[key])
        ax.grid(alpha=0.3)
        _style_axes(ax)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    # BFS and LW residuals are both in MHz^2, so they share one axis.
    ax = axes[len(keys)]
    unit, s_sq, _ = _residual_units("bfs")
    tops = []
    for key, colour, lbl in (("bfs", "tab:blue", "BFS"),
                             ("lw", "tab:orange", "LW")):
        m, s = _draw_dose_series(
            ax, all_results,
            lambda res, k=key: _folder_scan_residuals(res, k),
            f"mean d^2{_unit_sq(unit)}", colour, scale=s_sq,
            label=lbl, show_points=False)
        tops.append(np.nanmax(m + s))
    ax.set_ylim(0, max(tops) * 1.12)
    ax.set_title("Scatter about the tabletop vs dose", fontsize=13)
    ax.legend(loc="best", fontsize=11)

    for j in range(len(keys) + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{DRUG} dose series - summary", fontsize=16,
                 fontweight="bold")
    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1))
    if handles:
        fig.legend(handles, labels, title="dose", loc="lower center",
                   ncol=min(len(labels), 5), frameon=True)
    _save(fig, "summary")
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
            ax.annotate(f"media = {_fmt_y(p['baseline'], unit_y)}"
                        f"{' ' + unit_y if unit_y else ''}",
                        xy=(0.02, 0.06), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", color="dimgrey")
            width = p["x_right"] - p["x_left"]
            # Mean of the individual per-scan variances, matching the console
            # report - not the scatter about this pooled curve.
            unit, s_sq, _ = _residual_units(signal_key)
            mres = _folder_mean_residual(all_results[folder], signal_key)
            res_txt = ("" if mres is None else
                       f"mean d^2 = {_fmt_resid(mres * s_sq)}{_unit_sq(unit)}")
            # Average of the individual tabletops' peripheries, so this equals
            # the mean of the per-scan numbers on the grid panels - not the
            # periphery of this pooled curve.
            periph = mean_scan_periphery(all_results[folder],
                                         _periphery_region_key(signal_key))
            per_txt = ("" if periph is None
                       else f"   periphery = {periph:.1f}%")
            ax.set_title(f"{_flabel(folder)} (n={used})\n"
                         f"width = {width:.1f} µm   "
                         f"height = {_fmt_y(p['amplitude'], unit_y)}"
                         f"{' ' + unit_y if unit_y else ''}\n"
                         f"{res_txt}{per_txt}",
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
    _finish(fig, f"{signal_key}_tabletops", subdir=signal_key)


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
                  f"d^2 = {_fmt_resid(msq * s_sq):>9}"
                  f"  RMS = {_fmt_resid(np.sqrt(msq) * s_lin):>8}")
        if vals:
            v = np.array(vals)
            print(f"    {'-> mean over ' + str(len(v)) + ' scans':<40} "
                  f"          d^2 = {_fmt_resid(v.mean() * s_sq):>9}"
                  f"  RMS = {_fmt_resid(np.sqrt(v.mean()) * s_lin):>8}"
                  f"   (spread of d^2 +/- {_fmt_resid(v.std() * s_sq)})")
        else:
            print(f"    {'-> no usable scans':<40}")
    if unit:
        print(f"  (d^2 in {unit}^2, RMS in {unit})")
    else:
        print("  (dimensionless signal)")


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
    # Gradient panels share one y scale so doses can be compared by eye.
    ylim = _gradient_ylim(all_results, signal_key)
    for folder in FOLDERS:
        plot_folder_grid(all_results[folder], folder, signal_key, ncols=3,
                         ylim=ylim)
    for folder in FOLDERS:
        plot_folder_overlay(all_results[folder], folder,
                            FOLDER_COLORS[folder], signal_key, ylim=ylim)
    plot_all_stacked(all_results, signal_key)
    plot_folder_means(all_results, signal_key)
    if signal_key in TABLETOP_SIGNALS:
        report_tabletop_residuals(all_results, signal_key)
        plot_folder_tabletops(all_results, signal_key)
        plot_tabletop_dose_lines(all_results, signal_key)
    # The dose-series summary, regardless of SIGNAL_KIND.
    plot_summary(all_results)


if __name__ == "__main__":
    # 'bfs', 'lw', 'bfs_0', 'lw_0', 'vca', 'sobel'.
    SIGNAL_KIND = "lw_0"

    all_results = {}
    for folder in FOLDERS:
        path = folder_path(folder)
        print(f"Loading {path} ...")
        all_results[folder] = load_and_analyse(path)
        print(f"  {len(all_results[folder])} scans")

    report_vendor_agreement(all_results)
    make_all_plots(all_results, SIGNAL_KIND)
