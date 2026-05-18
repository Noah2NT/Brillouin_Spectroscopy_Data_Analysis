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
    _, shifts, linewidths, _ = bf.extract_fit_parameters(params)

    pixel_grad       = spectral_gradient_per_pixel(spectra_norm, "simple")
    pixel_grad_sobel = spectral_gradient_per_pixel(spectra_norm, "sobel")
    pixel_grad_lap   = spectral_gradient_per_pixel(spectra_norm, "lap")

    try:
        tab_params, tab_curve = bf.fit_tabletop(positions, shifts)
        region = bf.tabletop_regions(positions, tab_params, edge_pad=2.0)
    except Exception:
        tab_params, tab_curve, region = None, None, None

    return {"scan": scan, "shifts": shifts, "linewidths": linewidths,
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
            def _stats(arr):
                v = arr[grad_mask] if grad_mask.any() else np.array([np.nan])
                return (float(np.nanmean(v)), float(np.nanmin(v)),
                        float(np.nanmax(v)))
            mg, _, _ = _stats(result["pixel_grad"])
            ms, mns, mxs = _stats(result["pixel_grad_sobel"])
            ml, mnl, mxl = _stats(result["pixel_grad_lap"])
            out[r] = {
                "var":             float(np.var(residuals[mask])) * 10000.0,
                "mean_shift":      float(np.mean(result["shifts"][mask])),
                "mean_grad":       mg,
                "mean_grad_sobel": ms,
                "min_grad_sobel":  mns,
                "max_grad_sobel":  mxs,
                "mean_grad_lap":   ml,
                "min_grad_lap":    mnl,
                "max_grad_lap":    mxl,
            }
        else:
            out[r] = {"var": np.nan, "mean_shift": np.nan,
                      "mean_grad": np.nan,
                      "mean_grad_sobel": np.nan,
                      "min_grad_sobel": np.nan,
                      "max_grad_sobel": np.nan,
                      "mean_grad_lap": np.nan,
                      "min_grad_lap": np.nan,
                      "max_grad_lap": np.nan}
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

# vca_demo(ctrl_demo, "Control", num_endmembers=2)
# vca_demo(treat_demo, "Treated", num_endmembers=2)

# Frequency shift grids — every scan as a small line plot
plot_grid(ctrl_results,  "Control", ncols=5)
plot_grid(treat_results, "Treated", ncols=4)


# ─────────────────────────────────────────────
# Linewidth line graphs — every scan in each group
# ─────────────────────────────────────────────

def plot_linewidth_grid(results, group_label, ncols):
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        ax.plot(positions, r["linewidths"], 'b-', linewidth=1.2)
        ax.set_title(r["scan"]["filename"].rsplit(".csv", 1)[0], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label} — linewidth (Lorentzian gamma) "
                 f"along line (n={len(results)})", fontsize=13)
    fig.supxlabel("X position (um)", fontsize=10)
    fig.supylabel("Linewidth (GHz)", fontsize=10)
    plt.tight_layout()
    plt.show()


# plot_linewidth_grid(ctrl_results,  "Control", ncols=5)
# plot_linewidth_grid(treat_results, "Treated", ncols=4)


# ─────────────────────────────────────────────
# Per-pixel Lorentzian fits with shift + linewidth annotated
# ─────────────────────────────────────────────

def plot_per_pixel_lorentzians(result, group_label, ncols=10):
    scan = result["scan"]
    shift_freq = scan["shift_freq"]
    spectra = scan["folded_spectra_norm"]
    shifts = result["shifts"]
    linewidths = result["linewidths"]
    fitted_curves, _ = bf.fit_all_spectra(shift_freq, spectra)

    n = spectra.shape[0]
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.0, nrows * 1.7),
                             sharex=True)
    axes = np.atleast_2d(axes).flatten()
    for i in range(n):
        ax = axes[i]
        ax.plot(shift_freq, spectra[i], 'k-', linewidth=1.0)
        ax.plot(shift_freq, fitted_curves[i], 'r-', linewidth=1.0, alpha=0.8)
        ax.set_title(
            f"px {i}\nshift={shifts[i]:.3f}\nlw={linewidths[i]:.3f}",
            fontsize=6.5,
        )
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label}: per-pixel Lorentzian fits "
                 f"({scan['filename']}, {n} pixels)", fontsize=12)
    fig.supxlabel("Frequency Shift (GHz)", fontsize=9)
    fig.supylabel("Normalised intensity", fontsize=9)
    plt.tight_layout()
    plt.show()


# plot_per_pixel_lorentzians(ctrl_results[0], "Control", ncols=10)


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
# Excel export — one row per scan
# ─────────────────────────────────────────────

def build_export_row(result, group):
    fname = result["scan"]["filename"]
    if result["tab_params"] is None:
        return {"group": group, "filename": fname}

    p = result["tab_params"]
    s = region_stats(result)
    shifts = result["shifts"]
    linewidths = result["linewidths"]
    grad = result["pixel_grad"]
    grad_sobel = result["pixel_grad_sobel"]
    grad_lap = result["pixel_grad_lap"]
    region = result["region"]

    centre_mask = region == "center"
    edge_mask = region == "edge"
    outside_mask = region == "outside"
    n_centre = int(centre_mask.sum())
    n_edge = int(edge_mask.sum())
    cell_size = n_centre + n_edge
    edge_pct = (n_edge / cell_size) if cell_size > 0 else float("nan")

    # Slope normalised by cell width (smaller = sharper edges relative to size)
    cell_width = p["x_right"] - p["x_left"]
    slope_norm = p["slope_width"] / cell_width if cell_width > 0 else float("nan")

    if centre_mask.any():
        max_shift_c = float(np.nanmax(shifts[centre_mask]))
        min_shift_c = float(np.nanmin(shifts[centre_mask]))
    else:
        max_shift_c = min_shift_c = float("nan")

    # Linewidth stats per region (Lorentzian gamma)
    def _lw_stats(mask):
        if mask.any():
            v = linewidths[mask]
            return (float(np.nanmean(v)), float(np.nanmin(v)),
                    float(np.nanmax(v)))
        return (float("nan"), float("nan"), float("nan"))

    lw_c_mean, lw_c_min, lw_c_max = _lw_stats(centre_mask)
    lw_e_mean, lw_e_min, lw_e_max = _lw_stats(edge_mask)
    lw_o_mean, lw_o_min, lw_o_max = _lw_stats(outside_mask)

    # Sobel gradient applied to the 1D linewidth array
    n_pix = len(linewidths)
    lw_sobel = np.full(n_pix, np.nan)
    for i in range(2, n_pix - 2):
        lw_sobel[i] = abs(linewidths[i - 2] + 2.0 * linewidths[i - 1]
                          - 2.0 * linewidths[i + 1] - linewidths[i + 2])

    def _sobel_mean(mask):
        if mask.any():
            v = lw_sobel[mask]
            return float(np.nanmean(v))
        return float("nan")

    lw_sobel_overall = float(np.nanmean(lw_sobel))
    lw_sobel_c = _sobel_mean(centre_mask)
    lw_sobel_e = _sobel_mean(edge_mask)
    lw_sobel_o = _sobel_mean(outside_mask)

    return {
        "group":               group,
        "filename":            fname,
        "slope":               p["slope_width"],
        "slope_normalized":    slope_norm,
        "var_centre":          s["center"]["var"],
        "var_edge":            s["edge"]["var"],
        "shift_centre":        s["center"]["mean_shift"],
        "shift_edge":          s["edge"]["mean_shift"],
        "grad_centre":         s["center"]["mean_grad"],
        "grad_edge":           s["edge"]["mean_grad"],
        "grad_sobel_centre":      s["center"]["mean_grad_sobel"],
        "grad_sobel_edge":        s["edge"]["mean_grad_sobel"],
        "grad_sobel_min_centre":  s["center"]["min_grad_sobel"],
        "grad_sobel_max_centre":  s["center"]["max_grad_sobel"],
        "grad_sobel_min_edge":    s["edge"]["min_grad_sobel"],
        "grad_sobel_max_edge":    s["edge"]["max_grad_sobel"],
        "grad_lap_centre":        s["center"]["mean_grad_lap"],
        "grad_lap_edge":          s["edge"]["mean_grad_lap"],
        "grad_lap_min_centre":    s["center"]["min_grad_lap"],
        "grad_lap_max_centre":    s["center"]["max_grad_lap"],
        "grad_lap_min_edge":      s["edge"]["min_grad_lap"],
        "grad_lap_max_edge":      s["edge"]["max_grad_lap"],
        "max_shift_centre":    max_shift_c,
        "min_shift_centre":    min_shift_c,
        "min_shift":           float(np.nanmin(shifts)),
        "freq_shift_range":    float(np.nanmax(shifts) - np.nanmin(shifts)),
        "tabletop_range":      p["amplitude"],
        "lw_centre_mean":      lw_c_mean,
        "lw_centre_min":       lw_c_min,
        "lw_centre_max":       lw_c_max,
        "lw_edge_mean":        lw_e_mean,
        "lw_edge_min":         lw_e_min,
        "lw_edge_max":         lw_e_max,
        "lw_sobel_centre":     lw_sobel_c,
        "lw_sobel_edge":       lw_sobel_e,
        "tabletop_height":     p["baseline"] + p["amplitude"],
        "n_centre_pixels":     n_centre,
        "edge_pct":            edge_pct,
    }

rows = (
    [build_export_row(r, "Control") for r in ctrl_results]
    + [build_export_row(r, "Treated") for r in treat_results]
)
df_export = pd.DataFrame(rows)

excel_path = "line_scan_summary.xlsx"
df_export.to_excel(excel_path, index=False, sheet_name="line_scans")
print(f"\nExported {len(df_export)} rows to {excel_path}")


# ─────────────────────────────────────────────
# Separation score per feature: |mean_C - mean_T| / sqrt(std_C^2 + std_T^2)
# ─────────────────────────────────────────────

def separation_scores(df):
    ctrl = df[df["group"] == "Control"]
    treat = df[df["group"] == "Treated"]
    rows = []
    for col in df.columns:
        if col in ("group", "filename"):
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        m_c = float(np.nanmean(ctrl[col]))
        m_t = float(np.nanmean(treat[col]))
        s_c = float(np.nanstd(ctrl[col]))
        s_t = float(np.nanstd(treat[col]))
        denom = np.sqrt(s_c**2 + s_t**2)
        score = abs(m_c - m_t) / denom if denom > 0 else float("nan")
        rows.append({"feature": col, "score": score,
                     "mean_C": m_c, "std_C": s_c,
                     "mean_T": m_t, "std_T": s_t})
    sep = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return sep


sep = separation_scores(df_export)

print(f"\n=== Separation score: |mean_C - mean_T| / sqrt(std_C^2 + std_T^2) ===")
print(f"{'rank':>4}  {'feature':<28} {'score':>7}  "
      f"{'mean_C':>10} {'std_C':>9}  {'mean_T':>10} {'std_T':>9}")
print("-" * 86)
for i, row in sep.iterrows():
    print(f"{i+1:>4}  {row['feature']:<28} {row['score']:>7.3f}  "
          f"{row['mean_C']:>10.4f} {row['std_C']:>9.4f}  "
          f"{row['mean_T']:>10.4f} {row['std_T']:>9.4f}")

# Also save the ranking as a second sheet in the main Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    sep.to_excel(writer, index=False, sheet_name="separation")
print(f"Separation ranking written to {excel_path} (sheet: 'separation')")

# And also as its own standalone file (just feature + score)
sep_path = "separation_scores.xlsx"
sep[["feature", "score"]].to_excel(sep_path, index=False,
                                   sheet_name="separation")
print(f"Standalone ranking written to {sep_path}")


# ─────────────────────────────────────────────
# Tabletop fits for Control1 vs Treated1
#   row 1: linewidth + tabletop fit
#   row 2: VCA EM1 abundance + tabletop fit
# ─────────────────────────────────────────────

def find_result(results, name_substr):
    """Locate the analyse_scan result whose filename contains name_substr."""
    for r in results:
        if name_substr.lower() in r["scan"]["filename"].lower():
            return r
    raise ValueError(f"no result matching {name_substr!r}")


def fit_and_plot_with_tabletop(ax, scan, signal, title, ylabel):
    """Fit a tabletop to `signal` (1D over scan positions) and plot it."""
    positions = scan["positions"]
    try:
        tab_params, tab_curve = bf.fit_tabletop(positions, signal)
        region = bf.tabletop_regions(positions, tab_params, edge_pad=2.0)
    except Exception:
        tab_params, tab_curve, region = None, None, None

    shade_regions(ax, positions, region)
    ax.plot(positions, signal, 'k.', markersize=4, label='data')
    if tab_curve is not None:
        ax.plot(positions, tab_curve, 'r-', linewidth=1.8, label='tabletop fit')
        ax.axvline(tab_params["x_left"],  color='r', linestyle='--', alpha=0.5)
        ax.axvline(tab_params["x_right"], color='r', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("X position (um)")
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)


def vca_abundances_2em(scan):
    """VCA(2) on this scan; return (EM1_abundance, EM2_abundance) per pixel."""
    spec = scan["folded_spectra_norm"]
    endmembers, _ = bd.vca(spec, num_endmembers=2)
    ab = bd.estimate_abundances(spec, endmembers)
    return ab[:, 0], ab[:, 1]


# ── Manual override: which endmember "goes up" (low-high-low tabletop)
# for each scan. Keyed by filename substring (case-insensitive). Value is
# 0 for EM1 or 1 for EM2. Fill these in by inspecting the working-EM grid
# plots — anything not listed falls back to the auto-detector below.
EM_CHOICE = {
    # "control1.1_5um_9s":  0,
    # "control14":          1,
    # "treated_5um_9s":     0,
    # "6.25nM_1.1_6s_5um":  1,
}


def pick_working_em(positions, em1, em2, filename=""):
    """Return (signal, label). Manual override first, then auto-pick.

    Auto-pick uses centre-vs-edge mean of each endmember; whichever has the
    larger positive (centre − edge) value is the one "going up".
    """
    base = filename.rsplit(".csv", 1)[0].lower()
    for key, idx in EM_CHOICE.items():
        if key.lower() in base:
            return (em1, "EM1") if idx == 0 else (em2, "EM2")

    n = len(em1)
    mid_slice = slice(n // 3, 2 * n // 3)
    edge_idx = list(range(0, n // 5)) + list(range(4 * n // 5, n))
    score1 = float(np.nanmean(em1[mid_slice]) - np.nanmean(em1[edge_idx]))
    score2 = float(np.nanmean(em2[mid_slice]) - np.nanmean(em2[edge_idx]))
    return (em1, "EM1") if score1 >= score2 else (em2, "EM2")


def plot_raw_line(ax, scan, signal, title, ylabel):
    """Plain line plot — no tabletop, no region shading."""
    positions = scan["positions"]
    ax.plot(positions, signal, 'k-', linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("X position (um)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


ctrl1 = find_result(ctrl_results, "control1")
treat1 = find_result(treat_results, "treated1")

ctrl1_em1, ctrl1_em2 = vca_abundances_2em(ctrl1["scan"])
treat1_em1, treat1_em2 = vca_abundances_2em(treat1["scan"])

ctrl1_signal,  ctrl1_label  = pick_working_em(
    ctrl1["scan"]["positions"], ctrl1_em1, ctrl1_em2,
    ctrl1["scan"]["filename"])
treat1_signal, treat1_label = pick_working_em(
    treat1["scan"]["positions"], treat1_em1, treat1_em2,
    treat1["scan"]["filename"])

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Row 1: linewidth (raw line, no tabletop)
plot_raw_line(axes[0, 0], ctrl1["scan"], ctrl1["linewidths"],
              "Control", "Brillouin linewidth (GHz)")
plot_raw_line(axes[0, 1], treat1["scan"], treat1["linewidths"],
              "Treated", "Brillouin linewidth (GHz)")

# Row 2: working endmember abundance + tabletop fit
fit_and_plot_with_tabletop(axes[1, 0], ctrl1["scan"], ctrl1_signal,
                           f"Control ({ctrl1_label})",
                           "Working EM abundance")
fit_and_plot_with_tabletop(axes[1, 1], treat1["scan"], treat1_signal,
                           f"Treated ({treat1_label})",
                           "Working EM abundance")

plt.suptitle("Linewidth (raw) + working VCA endmember tabletop fit",
             fontsize=13)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# Grids: linewidth (raw line, no tabletop) for ALL Controls and ALL Treated
# ─────────────────────────────────────────────

def plot_linewidth_grid_raw(results, group_label, ncols):
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        ax.plot(positions, r["linewidths"], 'k-', linewidth=1.2)
        ax.set_title(r["scan"]["filename"].rsplit(".csv", 1)[0], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label} — linewidth (n={len(results)})",
                 fontsize=13)
    fig.supxlabel("X position (um)", fontsize=10)
    fig.supylabel("Brillouin linewidth (GHz)", fontsize=10)
    plt.tight_layout()
    plt.show()


plot_linewidth_grid_raw(ctrl_results,  "Control", ncols=5)
plot_linewidth_grid_raw(treat_results, "Treated", ncols=4)


# ─────────────────────────────────────────────
# Grids: working VCA endmember + tabletop fit for ALL Controls and Treated
# ─────────────────────────────────────────────

def compute_working_em(results):
    """Per-scan: pick working endmember and fit tabletop; return list of dicts."""
    out = []
    for r in results:
        positions = r["scan"]["positions"]
        em1, em2 = vca_abundances_2em(r["scan"])
        signal, em_label = pick_working_em(positions, em1, em2,
                                            r["scan"]["filename"])
        try:
            tab_params, tab_curve = bf.fit_tabletop(positions, signal)
            region = bf.tabletop_regions(positions, tab_params, edge_pad=2.0)
        except Exception:
            tab_params, tab_curve, region = None, None, None
        out.append({
            "scan": r["scan"], "signal": signal, "em_label": em_label,
            "tab_params": tab_params, "tab_curve": tab_curve,
            "region": region,
        })
    return out


def plot_working_em_grid(em_results, group_label, ncols):
    nrows = (len(em_results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(em_results):
        ax = axes[i]
        positions = r["scan"]["positions"]
        ax.plot(positions, r["signal"], 'k-', linewidth=1.2)
        title = (f"{r['scan']['filename'].rsplit('.csv', 1)[0]} "
                 f"({r['em_label']})")
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(em_results), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label} — working VCA endmember "
                 f"(n={len(em_results)})", fontsize=13)
    fig.supxlabel("X position (um)", fontsize=10)
    fig.supylabel("Working EM abundance", fontsize=10)
    plt.tight_layout()
    plt.show()


def print_working_em_tabletop(em_results, group_label):
    print(f"\n=== {group_label} — VCA working endmember tabletop fits ===")
    hdr = (f"{'filename':<40} {'EM':>4}  "
           f"{'floor':>8} {'roof':>8} {'amp':>8}  "
           f"{'x_left':>8} {'x_right':>8} {'slope':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in em_results:
        name = r["scan"]["filename"].rsplit(".csv", 1)[0]
        if r["tab_params"] is None:
            print(f"{name:<40} {r['em_label']:>4}  (fit failed)")
            continue
        p = r["tab_params"]
        floor = p["baseline"]
        roof = floor + p["amplitude"]
        print(f"{name:<40} {r['em_label']:>4}  "
              f"{floor:>8.4f} {roof:>8.4f} {p['amplitude']:>8.4f}  "
              f"{p['x_left']:>8.2f} {p['x_right']:>8.2f} "
              f"{p['slope_width']:>7.3f}")


ctrl_em_results  = compute_working_em(ctrl_results)
treat_em_results = compute_working_em(treat_results)

plot_working_em_grid(ctrl_em_results,  "Control", ncols=5)
plot_working_em_grid(treat_em_results, "Treated", ncols=4)

print_working_em_tabletop(ctrl_em_results,  "CONTROL")
print_working_em_tabletop(treat_em_results, "TREATED")


# ─────────────────────────────────────────────
# Sobel gradient grids — applied along position to linewidth + working EM
# ─────────────────────────────────────────────

def sobel_1d(signal):
    """1D Sobel-Feldman gradient [1, 2, 0, -2, -1] applied across positions.
    Returns |gradient| with NaN at the two boundary pixels on each end."""
    n = len(signal)
    out = np.full(n, np.nan)
    for i in range(2, n - 2):
        out[i] = abs(signal[i - 2] + 2.0 * signal[i - 1]
                     - 2.0 * signal[i + 1] - signal[i + 2])
    return out


def plot_sobel_grid(positions_list, signals, titles, group_label,
                    ncols, ylabel, suptitle):
    nrows = (len(signals) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for i, (pos, sig, ttl) in enumerate(zip(positions_list, signals, titles)):
        ax = axes[i]
        grad = sobel_1d(sig)
        ax.plot(pos, grad, 'k-', linewidth=1.2)
        ax.set_title(ttl, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(len(signals), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{group_label} — {suptitle} (n={len(signals)})",
                 fontsize=13)
    fig.supxlabel("X position (um)", fontsize=10)
    fig.supylabel(ylabel, fontsize=10)
    plt.tight_layout()
    plt.show()


# Linewidth Sobel
plot_sobel_grid(
    [r["scan"]["positions"] for r in ctrl_results],
    [r["linewidths"] for r in ctrl_results],
    [r["scan"]["filename"].rsplit(".csv", 1)[0] for r in ctrl_results],
    "Control", ncols=5, ylabel="|Sobel(linewidth)|",
    suptitle="Sobel gradient of linewidth",
)
plot_sobel_grid(
    [r["scan"]["positions"] for r in treat_results],
    [r["linewidths"] for r in treat_results],
    [r["scan"]["filename"].rsplit(".csv", 1)[0] for r in treat_results],
    "Treated", ncols=4, ylabel="|Sobel(linewidth)|",
    suptitle="Sobel gradient of linewidth",
)

# VCA working-endmember Sobel
plot_sobel_grid(
    [r["scan"]["positions"] for r in ctrl_em_results],
    [r["signal"] for r in ctrl_em_results],
    [f"{r['scan']['filename'].rsplit('.csv', 1)[0]} ({r['em_label']})"
     for r in ctrl_em_results],
    "Control", ncols=5, ylabel="|Sobel(EM abundance)|",
    suptitle="Sobel gradient of working VCA endmember",
)
plot_sobel_grid(
    [r["scan"]["positions"] for r in treat_em_results],
    [r["signal"] for r in treat_em_results],
    [f"{r['scan']['filename'].rsplit('.csv', 1)[0]} ({r['em_label']})"
     for r in treat_em_results],
    "Treated", ncols=4, ylabel="|Sobel(EM abundance)|",
    suptitle="Sobel gradient of working VCA endmember",
)