"""Turning a painted canvas into per-pixel material fractions.

Downsampling is done by exact area overlap rather than by picking a nearest
sub-cell, so a scan pixel straddling a boundary reports the true proportion of
each material inside its footprint.  Canvas and scan dimensions need not be
multiples of one another.

The confocal point spread function is applied *before* downsampling, on the
fine grid, so optical blur and geometric coverage mix in the right order.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

#: A Gaussian's FWHM is this many standard deviations.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def axis_overlap_weights(n_in, n_out):
    """Area-overlap weights mapping ``n_in`` input cells onto ``n_out`` output cells.

    Returns
    -------
    ndarray, shape (n_out, n_in)
        Row ``i`` holds the fraction of output cell ``i`` covered by each input
        cell.  Every row sums to 1.
    """
    n_in, n_out = int(n_in), int(n_out)
    in_edges = np.arange(n_in + 1, dtype=float) / n_in
    out_edges = np.arange(n_out + 1, dtype=float) / n_out

    lo = np.maximum(out_edges[:-1, None], in_edges[None, :-1])
    hi = np.minimum(out_edges[1:, None], in_edges[None, 1:])
    overlap = np.clip(hi - lo, 0.0, None)

    totals = overlap.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return overlap / totals


def one_hot_planes(index_map, n_materials):
    """Expand an index map into one float plane per material.

    Returns
    -------
    ndarray, shape (n_materials, height, width), float32
    """
    index_map = np.asarray(index_map)
    n_materials = int(n_materials)
    planes = np.zeros((n_materials, index_map.size), dtype=np.float32)
    flat = index_map.ravel()
    valid = flat < n_materials
    planes[flat[valid], np.flatnonzero(valid)] = 1.0
    return planes.reshape((n_materials,) + index_map.shape)


def apply_psf(planes, sigma_subcells):
    """Blur each material plane by the confocal point spread function.

    Blurring the *composition* rather than the finished spectra is both faster
    and physically equivalent, because the spectrum is linear in the material
    fractions.
    """
    if sigma_subcells is None or sigma_subcells <= 0:
        return planes
    out = np.empty_like(planes)
    for i in range(planes.shape[0]):
        out[i] = gaussian_filter(planes[i], sigma=float(sigma_subcells),
                                 mode="nearest")
    return out


def psf_sigma_in_subcells(psf_fwhm_um, um_per_subcell):
    """Convert a PSF width in microns to a Gaussian sigma in canvas sub-cells."""
    if psf_fwhm_um is None or psf_fwhm_um <= 0 or um_per_subcell <= 0:
        return 0.0
    return (float(psf_fwhm_um) / FWHM_PER_SIGMA) / float(um_per_subcell)


def fractions_from_drawing(drawing, n_materials, grid_w, grid_h,
                           psf_fwhm_um=0.0):
    """Per-scan-pixel material fractions for a painted drawing.

    Parameters
    ----------
    drawing : CellDrawing
    n_materials : int
        Size of the material library, so the fraction array has a stable width
        even for materials that were never painted.
    grid_w, grid_h : int
        Scan grid dimensions in pixels.
    psf_fwhm_um : float
        Confocal spot size (FWHM) in microns.  Zero disables the blur.

    Returns
    -------
    ndarray, shape (grid_h * grid_w, n_materials)
        Rows are ordered X-fastest then Y, matching the CSV row order that
        ``brillouin_preprocessing.reshape_to_map`` expects.  Each row sums to 1.

    Notes
    -----
    The vertical axis is flipped on the way out.  Canvas row 0 is at the top of
    the screen, but every figure in this project is drawn with
    ``imshow(..., origin="lower")``, which puts scan row 0 at the bottom.
    Without the flip, a cell painted near the top of the canvas would come out
    near the bottom of the user's own analysis plots.
    """
    planes = one_hot_planes(drawing.index_map, n_materials)

    sigma = psf_sigma_in_subcells(
        psf_fwhm_um, min(drawing.um_per_subcell_x, drawing.um_per_subcell_y))
    planes = apply_psf(planes, sigma)

    wy = axis_overlap_weights(drawing.height_px, int(grid_h))
    wx = axis_overlap_weights(drawing.width_px, int(grid_w))

    # (n_mat, H, W) -> (n_mat, grid_h, grid_w)
    coarse = np.einsum("yh,mhw,xw->myx", wy, planes.astype(np.float64), wx,
                       optimize=True)
    coarse = coarse[:, ::-1, :]  # screen-down to plot-up, see Notes

    fractions = coarse.reshape(int(n_materials), -1).T  # (n_pixels, n_mat)
    totals = fractions.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return fractions / totals


def dominant_material_map(fractions, grid_w, grid_h):
    """The most abundant material in each pixel, shaped as an image."""
    return np.argmax(fractions, axis=1).reshape(int(grid_h), int(grid_w))


def mixing_fraction_map(fractions, grid_w, grid_h):
    """How mixed each pixel is: 0 for pure, approaching 1 for an even blend."""
    purity = fractions.max(axis=1)
    return (1.0 - purity).reshape(int(grid_h), int(grid_w))
