import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def load_brillouin_csv(filepath):
    """
    Load a Brillouin scan CSV file.

    Returns:
        spectra : ndarray (N, B) — raw intensity spectra
        freq    : ndarray (B,)   — frequency axis (GHz)
        df      : DataFrame      — original dataframe (contains X, Y columns)
    """
    df = pd.read_csv(filepath)
    spectra = np.vstack(
        df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
    )
    start = df["startFrequency"].iloc[0]
    end   = df["endFrequency"].iloc[0]
    freq  = np.linspace(start, end, spectra.shape[1])
    return spectra, freq, df


def extract_frequency_window(spectra, freq, start_idx, end_idx):
    """
    Slice spectra and frequency axis to the Brillouin peak window.

    Parameters:
        spectra   : ndarray (N, B)
        freq      : ndarray (B,)
        start_idx : int — index of window start
        end_idx   : int — index of window end (exclusive)

    Returns:
        peak_spectra : ndarray (N, W)
        peak_freq    : ndarray (W,)
    """
    return spectra[:, start_idx:end_idx], freq[start_idx:end_idx]


def fold_brillouin_spectrum(spectra, freq, rayleigh_trim_ghz=1.5,
                            exclude_ranges=None):
    """
    Fold spectra around the Rayleigh center to stack Stokes and anti-Stokes peaks.

    Both sides are ordered from closest-to-center outward so they align
    correctly before averaging. The Rayleigh tail region is trimmed.

    Parameters:
        spectra            : ndarray (N, B) — raw intensity spectra
        freq               : ndarray (B,)   — frequency axis in GHz
        rayleigh_trim_ghz  : float — trim bins within this distance of center (GHz)
        exclude_ranges     : list of (float, float) or None
                             Frequency ranges (in raw GHz) to exclude. Bins within
                             each (low, high) pair are replaced with the local
                             baseline before folding, so they don't contaminate
                             the averaged spectrum.
                             Example: [(5.2, 5.8), (7.0, 7.5)]

    Returns:
        folded_spectra : ndarray (N, W) — averaged Stokes + anti-Stokes
        shift_freq     : ndarray (W,)   — frequency shift axis (GHz, positive)
    """
    spectra = spectra.copy()

    # Mask out excluded frequency ranges before folding
    if exclude_ranges is not None:
        for low, high in exclude_ranges:
            mask = (freq >= low) & (freq <= high)
            if mask.any():
                # Replace with per-spectrum baseline (median of non-excluded bins)
                good = ~mask
                baseline = np.median(spectra[:, good], axis=1, keepdims=True)
                spectra[:, mask] = baseline

    center_freq = (freq[0] + freq[-1]) / 2.0
    center_idx = np.argmin(np.abs(freq - center_freq))
    bin_width = np.abs(freq[1] - freq[0])

    # Both sides ordered: index 0 = closest to Rayleigh, increasing = further away
    stokes_from_center = spectra[:, :center_idx][:, ::-1]   # reverse so idx 0 = nearest center
    anti_stokes_from_center = spectra[:, center_idx:]        # idx 0 = nearest center

    # Truncate to equal length
    w = min(stokes_from_center.shape[1], anti_stokes_from_center.shape[1])
    stokes_from_center = stokes_from_center[:, :w]
    anti_stokes_from_center = anti_stokes_from_center[:, :w]

    # Average for improved SNR
    folded = (stokes_from_center + anti_stokes_from_center) / 2.0

    # Build shift axis: index i corresponds to (i+1) * bin_width GHz from Rayleigh
    shift_freq = np.arange(1, w + 1) * bin_width

    # Trim Rayleigh tail region
    trim_bins = int(rayleigh_trim_ghz / bin_width)
    folded_spectra = folded[:, trim_bins:]
    shift_freq = shift_freq[trim_bins:]

    return folded_spectra, shift_freq


def build_coordinate_arrays(df):
    """
    Extract sorted unique X and Y scan positions from the dataframe.

    Returns:
        x_vals : ndarray — unique x positions in ascending order
        y_vals : ndarray — unique y positions in ascending order
    """
    x_vals = np.sort(df["X"].unique())
    y_vals = np.sort(df["Y"].unique())
    return x_vals, y_vals


def reshape_to_map(data, y_vals, x_vals):
    """
    Reshape a flat per-pixel array into a 2D spatial map.

    Parameters:
        data   : ndarray (N,) — one value per scan pixel
        y_vals : array-like   — unique y positions
        x_vals : array-like   — unique x positions

    Returns:
        ndarray (len(y_vals), len(x_vals))
    """
    return data.reshape(len(y_vals), len(x_vals))


def normalize_spectra(spectra):
    """
    Min-max normalize each spectrum independently to [0, 1].

    Parameters:
        spectra : ndarray (N, B)

    Returns:
        ndarray (N, B) — normalized spectra
    """
    mins = spectra.min(axis=1, keepdims=True)
    maxs = spectra.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return (spectra - mins) / ranges


def spatial_smooth_spectra(spectra, y_vals, x_vals, sigma=1.0):
    """
    Spatially smooth spectra using a Gaussian kernel over the 2D scan grid.

    Each frequency bin is independently smoothed across the spatial dimensions,
    so neighboring pixels' spectra bleed into each other with Gaussian weights.

    Parameters:
        spectra : ndarray (N, B) — one spectrum per scan pixel
        y_vals  : array-like     — unique y positions
        x_vals  : array-like     — unique x positions
        sigma   : float          — Gaussian kernel width in pixels

    Returns:
        ndarray (N, B) — spatially smoothed spectra
    """
    ny, nx = len(y_vals), len(x_vals)
    B = spectra.shape[1]
    cube = spectra.reshape(ny, nx, B)
    smoothed_cube = np.empty_like(cube)
    for b in range(B):
        smoothed_cube[:, :, b] = gaussian_filter(cube[:, :, b], sigma=sigma)
    return smoothed_cube.reshape(ny * nx, B)
