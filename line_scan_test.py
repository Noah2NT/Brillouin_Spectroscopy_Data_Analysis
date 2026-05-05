"""
Line-scan analysis pipeline for Control vs Treated samples.

Steps per scan:
  1. Fold raw spectra to isolate the Brillouin peak.
  2. Sum-normalise each pixel's Brillouin curve.
  3. Compute a scalar 'spectral gradient' per pixel by applying the
     [1, 0, -1] kernel ACROSS pixels — i.e. for each pixel i, sum the
     absolute bin-by-bin difference between spectra[i+1] and spectra[i-1].
  4. Lorentzian-fit each pixel's Brillouin peak to get a frequency shift.
  5. Fit a tabletop curve to the shift profile along the scan and use it
     to label every pixel as outside / edge / center.
  6. Per region, report residual variance (×10000), mean shift, and mean
     spectral gradient.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import brillouin_decomposition as bd


REGION_COLORS = {"outside": "#dddddd", "edge": "#ffd9b3", "center": "#b3e6ff"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def spectral_gradient_per_pixel(spectra, kind="simple"):
    """
    Apply a 1D kernel ACROSS pixels (axis 0) and return a scalar
    'amount of change' per pixel: sum |kernel * spectra| over frequency bins.

    kind:
        'simple'  -> [1, -1]            forward difference
        'sobel'   -> [1, 2, 0, -2, -1]  Sobel (smoothed centred difference)
        'lap'     -> [1, -2, 1]         Laplacian (2nd derivative)
    """
    n = spectra.shape[0]
    out = np.full(n, np.nan)
    if kind == "simple":
        for i in range(n - 1):
            out[i] = np.sum(np.abs(spectra[i + 1] - spectra[i]))
    elif kind == "lap":
        for i in range(1, n - 1):
            r = spectra[i - 1] - 2.0 * spectra[i] + spectra[i + 1]
            out[i] = np.sum(np.abs(r))
    elif kind == "sobel":
        for i in range(2, n - 2):
            r = (spectra[i - 2] + 2.0 * spectra[i - 1]
                 - 2.0 * spectra[i + 1] - spectra[i + 2])
            out[i] = np.sum(np.abs(r))
    else:
        raise ValueError(f"unknown kernel kind: {kind}")
    return out


def analyse_scan(scan):
    """Fit Lorentzians, compute spectral gradients (3 kernels), fit tabletop, label regions."""
    positions = scan["positions"]
    spectra_norm = scan["folded_spectra_norm"]

    _, params = bf.fit_all_spectra(scan["shift_freq"], spectra_norm)
    _, shifts, _, _ = bf.extract_fit_parameters(params)

    pixel_grad       = spectral_gradient_per_pixel(spectra_norm, "simple")
    pixel_grad_sobel = spectral_gradient_per_pixel(spectra_norm, "sobel")
    pixel_grad_lap   = spectral_gradient_per_pixel(spectra_norm, "lap")

    try:
        tab_params, tab_curve = bf.fit_tabletop(positions, shifts)
        region = bf.tabletop_regions(positions, tab_params, edge_pad=2.0)
    except Exception:
        tab_params, tab_curve, region = None, None, None

    return {"scan": scan, "shifts": shifts,
            "pixel_grad": pixel_grad,
            "pixel_grad_sobel": pixel_grad_sobel,
            "pixel_grad_lap": pixel_grad_lap,
            "tab_params": tab_params, "tab_curve": tab_curve, "region": region}


def region_stats(result):
    """Per-region residual variance (×10000), mean shift, mean spectral gradient.

    For the gradient calculation, the edge region is expanded by 1 pixel on
    each side so the centre and outside means aren't contaminated by the
    edge transitions. Variance and mean shift still use the original regions.
    """
    if result["region"] is None:
        return None
    residuals = result["shifts"] - result["tab_curve"]
    region = result["region"]

    # Expand edge mask by 1 pixel on each side
    edge_orig = region == "edge"
    edge_grad = edge_orig.copy()
    edge_grad[:-1] |= edge_orig[1:]
    edge_grad[1:]  |= edge_orig[:-1]

    out = {}
    for r in ("center", "edge", "outside"):
        mask = region == r
        if r == "edge":
            grad_mask = edge_grad
        elif r == "center":
            grad_mask = (region == "center") & ~edge_grad
        else:  # outside
            grad_mask = (region == "outside") & ~edge_grad

        if mask.any():
            def _mean(arr):
                v = arr[grad_mask] if grad_mask.any() else np.array([np.nan])
                return float(np.nanmean(v))
            out[r] = {
                "var":              float(np.var(residuals[mask])) * 10000.0,
                "mean_shift":       float(np.mean(result["shifts"][mask])),
                "mean_grad":        _mean(result["pixel_grad"]),
                "mean_grad_sobel":  _mean(result["pixel_grad_sobel"]),
                "mean_grad_lap":    _mean(result["pixel_grad_lap"]),
            }
        else:
            out[r] = {"var": np.nan, "mean_shift": np.nan,
                      "mean_grad": np.nan, "mean_grad_sobel": np.nan,
                      "mean_grad_lap": np.nan}
    return out


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


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

def plot_simple(result, ax, title):
    """Plain line plot of the raw fitted Brillouin shift."""
    positions = result["scan"]["positions"]
    ax.plot(positions, result["shifts"], 'k-', linewidth=1.2)
    ax.set_xlabel('X position (um)')
    ax.set_ylabel('Brillouin shift (GHz)')
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_grid(results, group_label, ncols):
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        ax.plot(positions, r["shifts"], 'k-', linewidth=1.2)
        ax.set_title(r["scan"]["filename"].rsplit(".csv", 1)[0], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label} (n={len(results)})", fontsize=13)
    fig.supxlabel("X position (um)", fontsize=10)
    fig.supylabel("Brillouin shift (GHz)", fontsize=10)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(results, group_label):
    print(f"\n=== {group_label} (n={len(results)}) — variance ×10000 ===")
    hdr = (f"{'filename':<32} {'slope':>6} | "
           f"{'var(c)':>7} {'var(e)':>7} {'var(o)':>7} | "
           f"{'shift(c)':>9} {'shift(e)':>9} {'shift(o)':>9} | "
           f"{'grad(c)':>8} {'grad(e)':>8} {'grad(o)':>8}")
    print(hdr)
    print("-" * len(hdr))

    cols = {"slope": [], "var_c": [], "var_e": [], "var_o": [],
            "sh_c": [], "sh_e": [], "sh_o": [],
            "gr_c": [], "gr_e": [], "gr_o": []}

    for r in results:
        fname = r["scan"]["filename"].rsplit(".csv", 1)[0]
        if r["tab_params"] is None:
            print(f"{fname:<32}  (fit failed)")
            continue
        sw = r["tab_params"]["slope_width"]
        s = region_stats(r)
        cols["slope"].append(sw)
        cols["var_c"].append(s["center"]["var"])
        cols["var_e"].append(s["edge"]["var"])
        cols["var_o"].append(s["outside"]["var"])
        cols["sh_c"].append(s["center"]["mean_shift"])
        cols["sh_e"].append(s["edge"]["mean_shift"])
        cols["sh_o"].append(s["outside"]["mean_shift"])
        cols["gr_c"].append(s["center"]["mean_grad"])
        cols["gr_e"].append(s["edge"]["mean_grad"])
        cols["gr_o"].append(s["outside"]["mean_grad"])

        print(
            f"{fname:<32} {sw:>6.2f} | "
            f"{s['center']['var']:>7.2f} {s['edge']['var']:>7.2f} "
            f"{s['outside']['var']:>7.2f} | "
            f"{s['center']['mean_shift']:>9.3f} "
            f"{s['edge']['mean_shift']:>9.3f} "
            f"{s['outside']['mean_shift']:>9.3f} | "
            f"{s['center']['mean_grad']:>8.4f} "
            f"{s['edge']['mean_grad']:>8.4f} "
            f"{s['outside']['mean_grad']:>8.4f}"
        )

    print("-" * len(hdr))
    for stat_label, fn in (("MEAN", np.nanmean), ("STD", np.nanstd)):
        m = {k: fn(np.asarray(v, dtype=float)) for k, v in cols.items()}
        print(
            f"{'GROUP ' + stat_label:<32} {m['slope']:>6.2f} | "
            f"{m['var_c']:>7.2f} {m['var_e']:>7.2f} {m['var_o']:>7.2f} | "
            f"{m['sh_c']:>9.3f} {m['sh_e']:>9.3f} {m['sh_o']:>9.3f} | "
            f"{m['gr_c']:>8.4f} {m['gr_e']:>8.4f} {m['gr_o']:>8.4f}"
        )


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

control_scans, treated_scans = bp.load_control_treated_line_scans("Line_scan_data")
control = bp.sum_normalize_line_scans(
    bp.fold_line_scans(control_scans, max_shift_ghz=8))
treated = bp.sum_normalize_line_scans(
    bp.fold_line_scans(treated_scans, max_shift_ghz=8))
print(f"Loaded {len(control)} control, {len(treated)} treated scans")

print("Analysing Control...")
ctrl_results = [analyse_scan(s) for s in control]
print("Analysing Treated...")
treat_results = [analyse_scan(s) for s in treated]


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

# (shift line graphs, per-scan grids, and per-pixel-fit grids removed —
#  only VCA demo below is kept)


# ─────────────────────────────────────────────
# VCA with 2 endmembers — example heatmaps
# ─────────────────────────────────────────────

def vca_demo(scan, group_label, num_endmembers=2):
    spectra = scan["folded_spectra_norm"]
    shift_freq = scan["shift_freq"]
    positions = scan["positions"]

    endmembers, _ = bd.vca(spectra, num_endmembers=num_endmembers)
    abundances = bd.estimate_abundances(spectra, endmembers)   # (N, p)

    fig, axes = plt.subplots(num_endmembers + 1, 1,
                             figsize=(11, 1.5 + num_endmembers * 1.2),
                             gridspec_kw={"height_ratios":
                                          [3] + [1] * num_endmembers})

    # Top panel: endmember spectra
    ax_em = axes[0]
    for k in range(num_endmembers):
        ax_em.plot(shift_freq, endmembers[k], linewidth=1.5,
                   label=f"EM {k + 1}")
    ax_em.set_xlabel("Frequency Shift (GHz)")
    ax_em.set_ylabel("Intensity")
    ax_em.set_title(f"{group_label}: VCA endmember spectra "
                    f"({scan['filename']})")
    ax_em.legend(fontsize=9)
    ax_em.grid(alpha=0.3)

    # Following panels: 1D abundance heatmaps
    for k in range(num_endmembers):
        ax = axes[k + 1]
        im = ax.imshow(abundances[:, k][np.newaxis, :], aspect="auto",
                       cmap="viridis", vmin=0, vmax=1,
                       extent=[positions[0], positions[-1], 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(f"EM {k + 1}", rotation=0, ha="right", va="center")
        plt.colorbar(im, ax=ax, label="abundance")
        if k == num_endmembers - 1:
            ax.set_xlabel("X position (um)")

    plt.tight_layout()
    plt.show()


def find_scan(scans, name_substr):
    """Return the first scan whose filename contains name_substr (case-insensitive)."""
    for s in scans:
        if name_substr.lower() in s["filename"].lower():
            return s
    raise ValueError(f"no scan found matching {name_substr!r}")


ctrl_demo = find_scan(control, "control18")
treat_demo = find_scan(treated, "treated2")

vca_demo(ctrl_demo, "Control", num_endmembers=2)
vca_demo(treat_demo, "Treated", num_endmembers=2)


# ─────────────────────────────────────────────
# VCA-abundance gradient per region (centre / edge / outside)
# ─────────────────────────────────────────────

def abundance_gradient(em, kind):
    """Apply 1D kernel to the abundance signal; returns |gradient| (length n,
    NaN at boundary positions where the kernel doesn't fit)."""
    n = len(em)
    out = np.full(n, np.nan)
    if kind == "simple":   # [1, -1]
        for i in range(n - 1):
            out[i] = abs(em[i + 1] - em[i])
    elif kind == "lap":    # [1, -2, 1]
        for i in range(1, n - 1):
            out[i] = abs(em[i - 1] - 2.0 * em[i] + em[i + 1])
    elif kind == "sobel":  # [1, 2, 0, -2, -1]
        for i in range(2, n - 2):
            out[i] = abs(em[i - 2] + 2.0 * em[i - 1]
                         - 2.0 * em[i + 1] - em[i + 2])
    else:
        raise ValueError(f"unknown kernel kind: {kind}")
    return out


def vca_abundance_region_grad(result, kind, num_endmembers=2):
    """
    Run VCA on this scan and compute the |gradient| of EM-1 abundance along
    the position axis using the chosen kernel; return the mean within each
    region. Edge mask is expanded by 1 pixel each side.
    """
    if result["region"] is None:
        return None

    scan = result["scan"]
    spectra = scan["folded_spectra_norm"]
    endmembers, _ = bd.vca(spectra, num_endmembers=num_endmembers)
    abundances = bd.estimate_abundances(spectra, endmembers)
    em1 = abundances[:, 0]
    grad = abundance_gradient(em1, kind)

    region = result["region"]
    edge_orig = region == "edge"
    edge_grad = edge_orig.copy()
    edge_grad[:-1] |= edge_orig[1:]
    edge_grad[1:]  |= edge_orig[:-1]

    out = {}
    for r in ("center", "edge", "outside"):
        if r == "edge":
            mask = edge_grad
        elif r == "center":
            mask = (region == "center") & ~edge_grad
        else:
            mask = (region == "outside") & ~edge_grad
        out[r] = float(np.nanmean(grad[mask])) if mask.any() else float("nan")
    return out


def print_vca_region_table(results, group_label, kind):
    label = {"simple": "[1,-1]", "sobel": "Sobel", "lap": "Laplacian"}[kind]
    print(f"\n=== {group_label} — VCA(2) abundance |gradient| ({label}) per region ===")
    print(f"{'filename':<40} {'centre':>10} {'edge':>10} {'outside':>10}")
    print("-" * 76)
    cols = {"center": [], "edge": [], "outside": []}
    for r in results:
        fname = r["scan"]["filename"].rsplit(".csv", 1)[0]
        s = vca_abundance_region_grad(r, kind)
        if s is None:
            print(f"{fname:<40}  (no region info)")
            continue
        for reg in ("center", "edge", "outside"):
            cols[reg].append(s[reg])
        print(f"{fname:<40} "
              f"{s['center']:>10.4f} {s['edge']:>10.4f} {s['outside']:>10.4f}")
    print("-" * 76)
    def m(lst):
        return np.nanmean(np.array(lst, dtype=float)) if lst else float("nan")
    def sd(lst):
        return np.nanstd(np.array(lst, dtype=float)) if lst else float("nan")
    print(f"{'GROUP MEAN':<40} "
          f"{m(cols['center']):>10.4f} {m(cols['edge']):>10.4f} "
          f"{m(cols['outside']):>10.4f}")
    print(f"{'GROUP STD':<40} "
          f"{sd(cols['center']):>10.4f} {sd(cols['edge']):>10.4f} "
          f"{sd(cols['outside']):>10.4f}")


for kind in ("simple", "sobel", "lap"):
    print_vca_region_table(ctrl_results,  "CONTROL", kind)
    print_vca_region_table(treat_results, "TREATED", kind)

# Summary tables
print_summary(ctrl_results, "CONTROL")
print_summary(treat_results, "TREATED")


# ─────────────────────────────────────────────
# Excel export — one row per scan  (commented out)
# ─────────────────────────────────────────────

# def build_export_row(result, group):
#     fname = result["scan"]["filename"]
#     if result["tab_params"] is None:
#         return {"group": group, "filename": fname}
#
#     p = result["tab_params"]
#     s = region_stats(result)
#     shifts = result["shifts"]
#     grad = result["pixel_grad"]
#     grad_sobel = result["pixel_grad_sobel"]
#     grad_lap = result["pixel_grad_lap"]
#     region = result["region"]
#
#     centre_mask = region == "center"
#     edge_mask = region == "edge"
#     n_centre = int(centre_mask.sum())
#     n_edge = int(edge_mask.sum())
#     cell_size = n_centre + n_edge
#     edge_pct = (n_edge / cell_size) if cell_size > 0 else float("nan")
#
#     # Slope normalised by cell width (smaller = sharper edges relative to size)
#     cell_width = p["x_right"] - p["x_left"]
#     slope_norm = p["slope_width"] / cell_width if cell_width > 0 else float("nan")
#
#     if centre_mask.any():
#         max_shift_c = float(np.nanmax(shifts[centre_mask]))
#         min_shift_c = float(np.nanmin(shifts[centre_mask]))
#     else:
#         max_shift_c = min_shift_c = float("nan")
#
#     return {
#         "group":               group,
#         "filename":            fname,
#         "slope":               p["slope_width"],
#         "slope_normalized":    slope_norm,
#         "var_centre":          s["center"]["var"],
#         "var_edge":            s["edge"]["var"],
#         "var_outside":         s["outside"]["var"],
#         "shift_centre":        s["center"]["mean_shift"],
#         "shift_edge":          s["edge"]["mean_shift"],
#         "shift_outside":       s["outside"]["mean_shift"],
#         "grad_centre":         s["center"]["mean_grad"],
#         "grad_edge":           s["edge"]["mean_grad"],
#         "grad_outside":        s["outside"]["mean_grad"],
#         "grad_sobel_centre":   s["center"]["mean_grad_sobel"],
#         "grad_sobel_edge":     s["edge"]["mean_grad_sobel"],
#         "grad_sobel_outside":  s["outside"]["mean_grad_sobel"],
#         "grad_lap_centre":     s["center"]["mean_grad_lap"],
#         "grad_lap_edge":       s["edge"]["mean_grad_lap"],
#         "grad_lap_outside":    s["outside"]["mean_grad_lap"],
#         "max_shift":           float(np.nanmax(shifts)),
#         "min_shift":           float(np.nanmin(shifts)),
#         "max_shift_centre":    max_shift_c,
#         "min_shift_centre":    min_shift_c,
#         "max_grad":              float(np.nanmax(grad)),
#         "min_grad":              float(np.nanmin(grad)),
#         "overall_avg_grad":      float(np.nanmean(grad)),
#         "max_grad_sobel":        float(np.nanmax(grad_sobel)),
#         "min_grad_sobel":        float(np.nanmin(grad_sobel)),
#         "overall_avg_grad_sobel":float(np.nanmean(grad_sobel)),
#         "max_grad_lap":          float(np.nanmax(grad_lap)),
#         "min_grad_lap":          float(np.nanmin(grad_lap)),
#         "overall_avg_grad_lap":  float(np.nanmean(grad_lap)),
#         "overall_variance":    float(np.nanvar(shifts - result["tab_curve"])) * 10000.0,
#         "tabletop_height":     p["baseline"] + p["amplitude"],
#         "tabletop_floor":      p["baseline"],
#         "n_centre_pixels":     n_centre,
#         "n_edge_pixels":       n_edge,
#         "cell_size":           cell_size,
#         "edge_pct":            edge_pct,
#     }
#
#
# rows = (
#     [build_export_row(r, "Control") for r in ctrl_results]
#     + [build_export_row(r, "Treated") for r in treat_results]
# )
# df_export = pd.DataFrame(rows)
#
# excel_path = "line_scan_summary.xlsx"
# df_export.to_excel(excel_path, index=False, sheet_name="line_scans")
# print(f"\nExported {len(df_export)} rows to {excel_path}")
