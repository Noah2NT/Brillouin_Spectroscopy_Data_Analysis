"""Writing scans out in the format the rest of this project reads.

The target is the 13-column instrument table used by every CSV in ``data/``::

    X,Y,Z,Raw Data,Fitting Data,Half Width,Base,Indensity,Frequency,
    startFrequency,endFrequency,marker1,marker2

Details that are easy to get wrong and that downstream code depends on:

* ``Raw Data`` and ``Fitting Data`` are double-quoted comma-joined strings of
  equal length.  Raw values are integer counts.
* Rows run X-fastest then Y over a complete rectangular grid, because
  ``brillouin_preprocessing.reshape_to_map`` does a bare ``reshape``.
* The spaced ``Raw Data`` header is used, not the ``RawData`` variant, because
  ``brillouin_preprocessing.load_brillouin_csv`` has no alias handling.
* ``Fitting Data`` is zero outside the ``marker1``-``marker2`` window.
* ``Indensity`` is an integrated area and ``Half Width`` is an HWHM.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np

#: Column order, exactly as the instrument writes it.
COLUMNS = ["X", "Y", "Z", "Raw Data", "Fitting Data", "Half Width", "Base",
           "Indensity", "Frequency", "startFrequency", "endFrequency",
           "marker1", "marker2"]

#: Columns for the batch manifest.
MANIFEST_COLUMNS = [
    "csv_file", "truth_file", "drawing", "grid_w", "grid_h", "step_um",
    "exposure_s", "temperature_k", "psf_fwhm_um", "conserve_area",
    "bfs_jitter_mhz", "seed", "n_pixels", "median_peak_counts",
    "median_brillouin_counts", "median_fitted_bfs_ghz", "median_true_bfs_ghz",
    "median_abs_bfs_error_mhz", "median_abs_bfs_noise_error_mhz",
    "median_fitted_linewidth_ghz", "median_abs_linewidth_error_mhz", "generated_at",
]


def _join_ints(values):
    return ",".join(str(int(v)) for v in values)


def _join_floats(values):
    # Zeros are written bare, matching real files where everything outside the
    # marker window is exactly "0"; this also keeps files a sensible size.
    return ",".join("0" if v == 0 else format(float(v), ".10g") for v in values)


def write_scan_csv(path, result, instrument, line_terminator="\n"):
    """Write one scan as a 13-column instrument CSV.

    Parameters
    ----------
    path : str
    result : ScanResult
    instrument : Instrument
    line_terminator : str
        ``"\\n"`` matches ``data/line_scans_old``; ``"\\r\\n"`` matches the
        newer files in ``data/csv``.  Either is read fine by pandas.

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    start = instrument.start_frequency_ghz
    end = instrument.end_frequency_ghz
    m1 = instrument.marker1_ghz
    m2 = instrument.marker2_ghz

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator=line_terminator)
        w.writerow(COLUMNS)
        for i in range(result.n_pixels):
            area, centre, linewidth, base = result.fit_params[i]
            w.writerow([
                int(result.x_index[i]),
                int(result.y_index[i]),
                0,
                _join_ints(result.counts[i]),
                _join_floats(result.fitted_curves[i]),
                # The instrument column is a HALF width; ours is a full one.
                format(linewidth / 2.0, ".4f"),
                format(base, ".4f"),
                format(area, ".4f"),
                format(centre, ".4f"),
                format(start, ".10g"),
                format(end, ".10g"),
                repr(float(m1)),
                repr(float(m2)),
            ])
    return path


def write_truth_sidecar(path, result, instrument, drawing_name=""):
    """Write the noise-free ground truth beside a generated CSV.

    Everything an algorithm could want to be scored against, and nothing that
    a real instrument would ever produce -- which is exactly why it lives in a
    separate file rather than in the CSV's fitted columns.

    Contents
    --------
    ``fractions``      (n_pixels, n_materials) area fraction per material
    ``bfs_ghz``        true area-weighted Brillouin shift per pixel
    ``linewidth_ghz``  true area-weighted observed linewidth (FWHM) per pixel
    ``area``           true integrated peak area per pixel
    ``dominant``       index of the most abundant material per pixel
    ``purity``         that material's fraction; 1.0 means a pure pixel
    ``x_index``, ``y_index``   pixel coordinates, X-fastest row order
    ``fitted_*``       what the fit actually recovered, for convenience
    ``noisefree_*``    what the same fitter returns on the noise-free
                       spectrum.  Score a noisy fit against *these* to measure
                       noise tolerance, and against ``bfs_ghz`` and friends to
                       measure total error including the single-Lorentzian
                       model's own limitations.
    ``meta``           JSON blob of settings, instrument and material names
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = {
        "version": 1,
        "drawing": drawing_name,
        "material_names": result.material_names,
        "settings": result.settings.to_dict(),
        "instrument": instrument.to_dict(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Ground truth for a simulated Brillouin scan. Row order is "
            "X-fastest then Y, matching the CSV. Reshape any per-pixel array "
            "with .reshape(grid_h, grid_w)."
        ),
    }
    extra = {}
    for key in ("noisefree_bfs_ghz", "noisefree_linewidth_ghz",
                "noisefree_area", "noisefree_base"):
        if key in result.truth:
            extra[key] = np.asarray(result.truth[key], dtype=np.float64)

    np.savez_compressed(
        path,
        **extra,
        fractions=result.fractions.astype(np.float32),
        bfs_ghz=result.truth["bfs_ghz"].astype(np.float64),
        linewidth_ghz=result.truth["linewidth_ghz"].astype(np.float64),
        area=result.truth["area"].astype(np.float64),
        dominant=result.truth["dominant"].astype(np.int16),
        purity=result.truth["purity"].astype(np.float32),
        x_index=result.x_index.astype(np.int32),
        y_index=result.y_index.astype(np.int32),
        fitted_area=result.fit_params[:, 0],
        fitted_bfs_ghz=result.fit_params[:, 1],
        fitted_linewidth_ghz=result.fit_params[:, 2],
        fitted_base=result.fit_params[:, 3],
        meta=json.dumps(meta),
    )
    return path


def load_truth_sidecar(path):
    """Read a truth sidecar back into a plain dict."""
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            out[key] = data[key]
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"]))
    return out


def append_manifest(path, row):
    """Append one row to the batch manifest, writing the header if new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS,
                           extrasaction="ignore", lineterminator="\n")
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


def manifest_row(csv_path, truth_path, result, drawing_name=""):
    """Summarise one generated scan for the manifest."""
    s = result.settings
    peak = result.counts.max(axis=1)
    ref_lw = result.truth.get("noisefree_linewidth_ghz",
                              result.truth["linewidth_ghz"])
    # peak height = area / (pi * hwhm), and column 2 is a full width
    brillouin_peak = result.fit_params[:, 0] / (
        np.pi * np.maximum(result.fit_params[:, 2] / 2.0, 1e-6))
    return {
        "csv_file": os.path.basename(csv_path),
        "truth_file": os.path.basename(truth_path),
        "drawing": drawing_name,
        "grid_w": s.grid_w,
        "grid_h": s.grid_h,
        "step_um": s.step_um,
        "exposure_s": s.exposure_s,
        "temperature_k": s.temperature_k,
        "psf_fwhm_um": s.psf_fwhm_um,
        "conserve_area": s.conserve_area,
        "bfs_jitter_mhz": s.bfs_jitter_mhz,
        "seed": s.seed,
        "n_pixels": result.n_pixels,
        "median_peak_counts": float(np.median(peak)),
        "median_brillouin_counts": float(np.median(brillouin_peak)),
        "median_fitted_bfs_ghz": float(np.median(result.fit_params[:, 1])),
        "median_true_bfs_ghz": float(np.median(result.truth["bfs_ghz"])),
        "median_abs_bfs_error_mhz": float(np.median(np.abs(result.bfs_error_mhz()))),
        "median_abs_bfs_noise_error_mhz": float(
            np.median(np.abs(result.bfs_noise_error_mhz()))),
        "median_fitted_linewidth_ghz": float(np.median(result.fit_params[:, 2])),
        "median_abs_linewidth_error_mhz": float(
            np.median(np.abs(result.fit_params[:, 2] - ref_lw)) * 1000.0),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def default_stem(drawing_name, settings):
    """Filename stem in the style of the project's own scans.

    Mirrors names like
    ``Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv``.
    """
    span_x = settings.grid_w * settings.step_um
    span_y = settings.grid_h * settings.step_um
    base = (drawing_name or "sample").strip() or "sample"
    return (f"{base}_xy_scanning{_num(span_x)}um_{_num(span_y)}um"
            f"_step{_num(settings.step_um)}um_{_num(settings.exposure_s)}s")


def _num(value):
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"
